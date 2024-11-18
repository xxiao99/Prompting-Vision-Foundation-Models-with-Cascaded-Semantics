

class Dual_model(nn.Module):
    def __init__(self, args):
        super(Dual_model, self).__init__()
        self.vit_base = VisionTransformer("sup_vitb16_imagenet21k", 224, num_classes=-1, vis=False)
        self.vit_base.load_from(np.load(os.path.join("./backbone_ckpt", "imagenet21k_ViT-B_16.npz")))

        for k, p in self.vit_base.named_parameters():
            if "ppt" in k or "cross_conv" in k or "deep_ppt" in k:
                p.requires_grad = True
            else:
                p.requires_grad = False
        

        self.low_level_prompt = LowLevelPrompt()
        

        self.reweight_adapter = ReWeightingAdapter(input_dim=768, num_classes=args.nb_classes)

    def forward(self, x):

        low_level_prompt = self.low_level_prompt(x)  

        x, p = self.vit_base(x, low_level_prompt=low_level_prompt)  
        return self.reweight_adapter(x), self.reweight_adapter(p)


class LowLevelPrompt(nn.Module):
    def __init__(self):
        super(LowLevelPrompt, self).__init__()

        self.fc_color = nn.Linear(256, 128)
        self.fc_texture = nn.Linear(256, 128)
        self.fc_shape = nn.Linear(256, 128)

    def forward(self, x):

        color_hist = self.extract_color_histogram(x)
        color_feat = self.fc_color(color_hist)


        texture = self.extract_gabor_texture(x)
        texture_feat = self.fc_texture(texture)


        shape = self.extract_sobel_shape(x)
        shape_feat = self.fc_shape(shape)


        return torch.cat([color_feat, texture_feat, shape_feat], dim=-1)

    def extract_color_histogram(self, x):

        batch_size = x.size(0)
        histograms = []
        for i in range(batch_size):
            img = x[i].cpu().permute(1, 2, 0).numpy() 
            hist = cv2.calcHist([img.astype(np.uint8)], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            histograms.append(torch.tensor(hist.flatten(), dtype=torch.float32))
        return torch.stack(histograms).to(x.device)

    def extract_gabor_texture(self, x):

        batch_size = x.size(0)
        textures = []
        gabor_kernels = self.create_gabor_kernels()
        for i in range(batch_size):
            img = x[i].mean(0).cpu().numpy()  
            texture_features = []
            for kernel in gabor_kernels:
                filtered_img = cv2.filter2D(img, cv2.CV_32F, kernel)
                texture_features.append(filtered_img.mean())  
            textures.append(torch.tensor(texture_features, dtype=torch.float32))
        return torch.stack(textures).to(x.device)

    def extract_sobel_shape(self, x):

        batch_size = x.size(0)
        shapes = []
        for i in range(batch_size):
            img = x[i].mean(0).cpu().numpy() 
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2).mean()  
            shapes.append(torch.tensor(edge_magnitude, dtype=torch.float32))
        return torch.stack(shapes).to(x.device)

    def create_gabor_kernels(self):

        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kernels.append(kernel)
        return kernels


class ReWeightingAdapter(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ReWeightingAdapter, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        weights = self.softmax(self.fc1(x))  
        return self.fc2(x * weights)  


def get_args():
    parser = argparse.ArgumentParser('Semantic Prompt Tuning Script for Image Classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--nb_classes', default=200, type=int)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--data_path', default='/path/to/dataset', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output', type=str, help='path to save checkpoints')
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size, drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=args.batch_size, drop_last=False
    )

    model = Dual_model(args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, model)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.epochs):
        train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, None
        )
        evaluate(data_loader_val, model, device)

    total_time = time.time() - start_time
    print(f'Training time: {datetime.timedelta(seconds=int(total_time))}')


if __name__ == '__main__':
    args = get_args()
    main(args)