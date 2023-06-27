class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)  #mlp linear projecton of flattened patches   [3(输入),768（输出）, kernel_size[16,16], stride[16,16]]   []
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x): #【8，3，224，224】
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)   #【8（batch size），768，14，14】----从（第三个）维度2后展平【8，768，196】----【8(batch)，196(token_num)，768(feat_num)】
        x = self.norm(x) #默认为none identity
        return x