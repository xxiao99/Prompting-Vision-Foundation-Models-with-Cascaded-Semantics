class CrossAttention(nn.Module):
    def __init__(self, dim=768, heads=12, dim_head=64, dropout=0.1):
        super(CrossAttention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.to_drop = nn.Dropout(dropout)

    def forward(self, x_qkv, y_q):
        b, n, _, h = *x_qkv.shape, self.heads
        k = self.to_k(x_qkv).view(b, n, h, -1).permute(0, 2, 1, 3)
        v = self.to_v(x_qkv).view(b, n, h, -1).permute(0, 2, 1, 3)
        q = self.to_q(y_q).view(b, 1, h, -1).permute(0, 2, 1, 3)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, -1)
        return self.to_out(out)


class Cross_Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super(Cross_Block, self).__init__()
        self.attention = CrossAttention(dim, heads, dim // heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, kv, q):
        kv = self.attn_norm(kv)
        q = self.attn_norm(q)
        x = self.attention(kv, q) + q
        x = self.ffn(self.ffn_norm(x)) + x
        return x


class EncoderWithPrompts(nn.Module):
    def __init__(self, config):
        super(EncoderWithPrompts, self).__init__()
        self.layers = nn.ModuleList(
            [Cross_Block(config.hidden_size, config.num_heads, config.mlp_dim) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.prompt_projectors = nn.ModuleList(
            [nn.Linear(config.prompt_dim, config.hidden_size) for _ in range(config.num_layers)]
        )
        self.skip_projectors = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.prompt_dim) for _ in range(config.num_layers)]
        )

    def forward(self, x, prompts):
        skip_prompts = prompts
        for layer, projector, skip_projector in zip(self.layers, self.prompt_projectors, self.skip_projectors):
            prompts_projected = projector(prompts)
            x = layer(x, prompts_projected)
            skip_prompts += skip_projector(x)
        return self.norm(x)


class SemanticPrompts(nn.Module):
    def __init__(self, config):
        super(SemanticPrompts, self).__init__()
        self.color_extractor = nn.Linear(config.color_dim, config.prompt_dim)
        self.texture_extractor = nn.Linear(config.texture_dim, config.prompt_dim)
        self.shape_extractor = nn.Linear(config.shape_dim, config.prompt_dim)
        self.merge = nn.Linear(config.prompt_dim * 3, config.prompt_dim)

    def forward(self, color_features, texture_features, shape_features):
        color_prompt = self.color_extractor(color_features)
        texture_prompt = self.texture_extractor(texture_features)
        shape_prompt = self.shape_extractor(shape_features)
        combined = torch.cat([color_prompt, texture_prompt, shape_prompt], dim=-1)
        return self.merge(combined)


class VisionTransformerWithPrompts(nn.Module):
    def __init__(self, config):
        super(VisionTransformerWithPrompts, self).__init__()
        self.encoder = EncoderWithPrompts(config)
        self.head = nn.Linear(config.hidden_size, config.num_classes)
        self.semantic_prompt_module = SemanticPrompts(config)

    def forward(self, x, color_features, texture_features, shape_features):
        prompts = self.semantic_prompt_module(color_features, texture_features, shape_features)
        x = self.encoder(x, prompts)
        return self.head(x[:, 0])


class Config:
    hidden_size = 768
    num_heads = 12
    mlp_dim = 3072
    num_layers = 12
    prompt_dim = 64
    color_dim = 128
    texture_dim = 128
    shape_dim = 128
    num_classes = 1000


if __name__ == "__main__":
    config = Config()
    model = VisionTransformerWithPrompts(config)

    image_tokens = torch.randn(32, 197, config.hidden_size)
    color_features = torch.randn(32, config.color_dim)
    texture_features = torch.randn(32, config.texture_dim)
    shape_features = torch.randn(32, config.shape_dim)

    logits = model(image_tokens, color_features, texture_features, shape_features)
    print(logits.shape)