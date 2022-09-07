import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=4):
        super().__init__()

        self.num_heads = num_heads
        self.dk = (embedding_dim // num_heads) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):  # input: [B, emb_dim / 2, emb_dim]
        qkv = self.qkv_layer(x)  # out: [B, emb_dim, emb_dim * 3]

        # out: k=3 * [B, num_heads, emb_dim / 2, emb_dim / num_heads]
        query, key, value = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.num_heads))
        # same: torch.matmul(query, key.transpose(3,2)) ? dk, out: [B, num_heads, emb_dim / 2, emb_dim / 2]
        qk_mul = torch.einsum("... i d , ... j d -> ... i j", query, key) / self.dk
        if mask is not None:  # mask padding
            qk_mul.masked_fill_(mask, -np.inf)

        attention = torch.softmax(qk_mul, dim=-1)
        # out: [B, num_heads, emb_dim / 2, emb_dim / num_heads]
        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")  # out: [B, emb_dim / 2, (emb_dim / num_heads) * num_heads]
        x = self.out_attention(x)  # out: [B, emb_dim / 2, emb_dim]

        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, dropout=0.1):
        super(MultiLayerPerceptron, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(embedding_dim, num_heads)
        self.mlp = MultiLayerPerceptron(embedding_dim, mlp_dim)
        self.norm_layer1 = nn.LayerNorm(embedding_dim)
        self.norm_layer2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x_att = self.mha(x)
        x_att = self.dropout(x_att)
        x = x + x_att
        x = self.norm_layer1(x)

        x_mlp = self.mlp(x)
        x = x + x_mlp
        x = self.norm_layer2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim, num_blocks=12):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(embedding_dim, num_heads, mlp_dim) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.encoder(x)
        return x


class ViT(nn.Module):
    def __init__(self, img_size, in_channels, embedding_dim, num_heads, mlp_dim,
                 block_num, patch_dim, classification=True, num_classes=1, dropout=0.1):
        super(ViT, self).__init__()

        self.patch_dim = patch_dim  # 16
        self.classification = classification
        self.num_tokens = (img_size // patch_dim) ** 2  # 224 // 16 = 14 -> 14^2 = 196 image tokens
        self.token_dim = in_channels * (patch_dim ** 2)  # 3 * (16^2) = 768

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(p=dropout)

        self.transformer = Encoder(embedding_dim, num_heads, mlp_dim, block_num)

        # if self.classification:
        self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(x, 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)', patch_x=self.patch_dim,
                                patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape

        project = self.projection(img_patches)
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...', batch_size=batch_size)

        patches = torch.cat([token, project], dim=1)
        patches += self.pos_embedding[:tokens + 1, :]

        x = self.dropout(patches)
        x = self.transformer(x)
        # x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]
        x_cls = self.mlp_head(x[:, 0, :])
        x = x[:, 1:, :]

        return x, x_cls


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super(EncoderBottleneck, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DecoderBottleneck, self).__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # self.upsample = nn.Sequential(
        #     nn.ConvTranspose2d(_, _, kernel_size=2, stride=scale_factor),
        #     nn.ReLU(inplace=True),
        # )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_=None):
        x = self.upsample(x)

        if x_ is not None:
            x = torch.cat([x_, x], dim=1)

        x = self.conv_block(x)
        return x


class UnetEncoder(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, num_heads, mlp_dim,
                 block_num, patch_dim, embedding_dim=512, num_classes=3):
        super(UnetEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_image_size = image_size // patch_dim
        self.vit = ViT(self.vit_image_size, out_channels * 8, out_channels * 8, num_heads, mlp_dim,
                       block_num, patch_dim=1, classification=False, num_classes=num_classes)

        self.conv2 = nn.Conv2d(out_channels * 8, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)

        x, x_cls = self.vit(x)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_image_size, y=self.vit_image_size)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x, x1, x2, x3, x_cls


class UnetDecoder(nn.Module):
    def __init__(self, out_channels, num_classes):
        super(UnetDecoder, self).__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), num_classes, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x


class TransUNET(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, num_heads, mlp_dim, block_num, patch_dim, num_classes):
        super(TransUNET, self).__init__()

        self.encoder = UnetEncoder(image_size, in_channels, out_channels, num_heads, mlp_dim,
                                   block_num, patch_dim, num_classes=num_classes)
        self.decoder = UnetDecoder(out_channels, num_classes=1)  # num_classes

    def forward(self, x):
        x, x1, x2, x3, x_cls = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)

        return x, x_cls


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, preds, target):

        ce_loss = F.cross_entropy(preds, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


if __name__ == "__main__":
    model = TransUNET(128, 3, 128, 4, 512, 8, 16, 1)
    input_tensor = torch.randn(8, 3, 128, 128)
    model.eval()
    output_tensor, _ = model(input_tensor)
    print(output_tensor.shape)
