import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch import nn
import cv2
from super_image import EdsrModel, ImageLoader

device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device}')

class _conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(_conv, self).__init__(in_channels = in_channels, out_channels = out_channels, 
                               kernel_size = kernel_size, stride = stride, padding = (kernel_size) // 2, bias = True)
        
        self.weight.data = torch.normal(torch.zeros((out_channels, in_channels, kernel_size, kernel_size)), 0.02)
        self.bias.data = torch.zeros((out_channels))
        
        for p in self.parameters():
            p.requires_grad = True
        

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, BN = False, act = None, stride = 1, bias = True):
        super(conv, self).__init__()
        m = []
        m.append(_conv(in_channels = in_channel, out_channels = out_channel, 
                               kernel_size = kernel_size, stride = stride, padding = (kernel_size) // 2, bias = True))
        
        if BN:
            m.append(nn.BatchNorm2d(num_features = out_channel))
        
        if act is not None:
            m.append(act)
        
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        out = self.body(x)
        return out
        
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, act = nn.ReLU(inplace = True), bias = True):
        super(ResBlock, self).__init__()
        m = []
        m.append(conv(channels, channels, kernel_size, BN = True, act = act))
        m.append(conv(channels, channels, kernel_size, BN = True, act = None))
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_res_block, act = nn.ReLU(inplace = True)):
        super(BasicBlock, self).__init__()
        m = []
        
        self.conv = conv(in_channels, out_channels, kernel_size, BN = False, act = act)
        for i in range(num_res_block):
            m.append(ResBlock(out_channels, kernel_size, act))
        
        m.append(conv(out_channels, out_channels, kernel_size, BN = True, act = None))
        
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        res = self.conv(x)
        out = self.body(res)
        out += res
        
        return out
        
class Upsampler(nn.Module):
    def __init__(self, channel, kernel_size, scale, act = nn.ReLU(inplace = True)):
        super(Upsampler, self).__init__()
        m = []
        m.append(conv(channel, channel * scale * scale, kernel_size))
        m.append(nn.PixelShuffle(scale))
    
        if act is not None:
            m.append(act)
        
        self.body = nn.Sequential(*m)
    
    def forward(self, x):
        out = self.body(x)
        return out

class discrim_block(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, act = nn.LeakyReLU(inplace = True)):
        super(discrim_block, self).__init__()
        m = []
        m.append(conv(in_feats, out_feats, kernel_size, BN = True, act = act))
        m.append(conv(out_feats, out_feats, kernel_size, BN = True, act = act, stride = 2))
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        out = self.body(x)
        return out

class MiniSRGAN(nn.Module):
    
    def __init__(self, img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 8, act = nn.PReLU(), scale=4):
        super(MiniSRGAN, self).__init__()
        
        self.conv01 = conv(in_channel = img_feat, out_channel = n_feats, kernel_size = 9, BN = False, act = act)
        
        resblocks = [ResBlock(channels = n_feats, kernel_size = 3, act = act) for _ in range(num_block)]
        self.body = nn.Sequential(*resblocks)
        
        self.conv02 = conv(in_channel = n_feats, out_channel = n_feats, kernel_size = 3, BN = True, act = None)
        
        if(scale == 4):
            upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = 2, act = act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = scale, act = act)]

        self.tail = nn.Sequential(*upsample_blocks)
        
        self.last_conv = conv(in_channel = n_feats, out_channel = img_feat, kernel_size = 3, BN = False, act = nn.Tanh())
        
    def forward(self, x):
        
        x = self.conv01(x)
        _skip_connection = x
        
        x = self.body(x)
        x = self.conv02(x)
        feat = x + _skip_connection
        
        x = self.tail(feat)
        x = self.last_conv(x)
        
        return x, feat

class TinySRGAN(nn.Module):
    
    def __init__(self, img_feat = 3, n_feats = 32, kernel_size = 3, num_block = 6, act = nn.PReLU(), scale=4):
        super(TinySRGAN, self).__init__()
        
        self.conv01 = conv(in_channel = img_feat, out_channel = n_feats, kernel_size = 9, BN = False, act = act)
        
        resblocks = [ResBlock(channels = n_feats, kernel_size = 3, act = act) for _ in range(num_block)]
        self.body = nn.Sequential(*resblocks)
        
        self.conv02 = conv(in_channel = n_feats, out_channel = n_feats, kernel_size = 3, BN = True, act = None)
        
        if(scale == 4):
            upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = 2, act = act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(channel = n_feats, kernel_size = 3, scale = scale, act = act)]

        self.tail = nn.Sequential(*upsample_blocks)
        
        self.last_conv = conv(in_channel = n_feats, out_channel = img_feat, kernel_size = 3, BN = False, act = nn.Tanh())
        
    def forward(self, x):
        
        x = self.conv01(x)
        _skip_connection = x
        
        x = self.body(x)
        x = self.conv02(x)
        feat = x + _skip_connection
        
        x = self.tail(feat)
        x = self.last_conv(x)
        
        return x, feat


def build_generator():
    
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, expansion=6, stride=1, alpha=1.0):
            super(ResidualBlock, self).__init__()
            self.expansion = expansion
            self.stride = stride
            self.in_channels = in_channels
            self.out_channels = int(out_channels * alpha)
            self.pointwise_conv_filters = self._make_divisible(self.out_channels, 8)
            self.conv1 = nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn1 = nn.BatchNorm2d(in_channels * expansion)
            self.conv2 = nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, stride=stride, padding=1, groups=in_channels * expansion, bias=True)
            self.bn2 = nn.BatchNorm2d(in_channels * expansion)
            self.conv3 = nn.Conv2d(in_channels * expansion, self.pointwise_conv_filters, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn3 = nn.BatchNorm2d(self.pointwise_conv_filters)
            self.relu = nn.ReLU(inplace=True)
            self.skip_add = (stride == 1 and in_channels == self.pointwise_conv_filters)

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.skip_add:
                out = out + identity

            return out

        @staticmethod
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

    class Generator(nn.Module):
        def __init__(self, in_channels, num_residual_blocks, gf):
            super(Generator, self).__init__()
            self.num_residual_blocks = num_residual_blocks
            self.gf = gf

            self.conv1 = nn.Conv2d(in_channels, gf, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(gf)
            self.prelu1 = nn.PReLU()

            self.residual_blocks = self.make_layer(ResidualBlock, gf, num_residual_blocks)

            self.conv2 = nn.Conv2d(gf, gf, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(gf)

            self.upsample1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(gf, gf, kernel_size=3, stride=1, padding=1),
                nn.PReLU()
            )

            self.upsample2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(gf, gf, kernel_size=3, stride=1, padding=1),
                nn.PReLU()
            )

            self.conv3 = nn.Conv2d(gf, 3, kernel_size=3, stride=1, padding=1)
            self.tanh = nn.Tanh()

        def make_layer(self, block, out_channels, blocks):
            layers = []
            for _ in range(blocks):
                layers.append(block(out_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            out1 = self.prelu1(self.bn1(self.conv1(x)))
            out = self.residual_blocks(out1)
            out = self.bn2(self.conv2(out))
            out = out + out1
            out = self.upsample1(out)
            out = self.upsample2(out)
            out = self.tanh(self.conv3(out))
            return out

    return Generator(3, 6, 32)


def numpify(imgs):
    all_images = []
    for img in imgs:
        img = img.permute(1,2,0).to('cpu') ### MIGHT CRASH HERE
        all_images.append(img)
    return np.stack(all_images, axis=0)

transform = transforms.Compose([
            transforms.ToTensor()
        ])


# Function to translate the image
def translate_image(image, sharpen, model_name, save):
    print('Translating!')

    desired_width = 480
    
    original_width, original_height = image.size
    desired_height = int((original_height / original_width) * desired_width)

    resized_image = image.resize((desired_width, desired_height))

    if(model_name=='MobileSR'):
        
        model=build_generator().to(device)
        model.load_state_dict(torch.load('./weights/mobile_sr.pt'))

        low_res = transform(resized_image)
        low_res = low_res.unsqueeze(dim=0).to(device)
        model.eval()
        with torch.no_grad():
            sr = model(low_res)
            
        fake_imgs = numpify(sr)
        
        sr_img = Image.fromarray((((fake_imgs[0] + 1) / 2) * 255).astype(np.uint8))

    elif(model_name=='EDSR'):
        model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)    
        inputs = ImageLoader.load_image(resized_image)
        with torch.no_grad():
            preds = model(inputs)

        sr_img = preds.data.cpu().numpy()
        sr_img = sr_img[0].transpose((1, 2, 0)) * 255.0
        sr_img = Image.fromarray(sr_img.astype(np.uint8))
    elif(model_name=='MiniSRGAN'):
        model = MiniSRGAN().to(device)
        model.load_state_dict(torch.load('./weights/miniSRGAN.pt'))
        model.eval()
        
        inputs = np.array(resized_image)
        inputs = (inputs / 127.5) - 1.0   
        inputs = torch.tensor(inputs.transpose(2, 0, 1).astype(np.float32)).to(device)
        
        with torch.no_grad():
            output, _ = model(torch.unsqueeze(inputs,dim=0))
        output = output[0].cpu().numpy()
        output = np.clip(output, -1.0, 1.0)
        output = (output + 1.0) / 2.0
        output = output.transpose(1, 2, 0)
        sr_img = Image.fromarray((output * 255.0).astype(np.uint8))
        
    elif(model_name=='TinySRGAN'):
        model = TinySRGAN().to(device)
        model.load_state_dict(torch.load('./weights/tinySRGAN.pt'))
        
        inputs = np.array(resized_image)
        inputs = (inputs / 127.5) - 1.0   
        inputs = torch.tensor(inputs.transpose(2, 0, 1).astype(np.float32)).to(device)
        model.eval()
        
        with torch.no_grad():
            output, _ = model(torch.unsqueeze(inputs,dim=0))
        output = output[0].cpu().numpy()
        output = (output + 1.0) / 2.0
        output = output.transpose(1, 2, 0)
        sr_img = Image.fromarray((output * 255.0).astype(np.uint8))
    
    if sharpen:
        sr_img_cv = np.array(sr_img)
        sr_img_cv = cv2.cvtColor(sr_img_cv, cv2.COLOR_RGB2BGR)
        
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_sr_img_cv = cv2.filter2D(sr_img_cv, -1, kernel)
        
        sharpened_sr_img = Image.fromarray(cv2.cvtColor(sharpened_sr_img_cv, cv2.COLOR_BGR2RGB))

        if(save=="True"):
            sharpened_sr_img.save('super_resolved_image.png')
        
        return sharpened_sr_img
    else:
        
        if(save=="True"):
            sr_img.save('super_resolved_image.png')
        
        return sr_img
    
interface = gr.Interface(
    fn=translate_image,
    inputs=[
        gr.Image(type="pil"),
        gr.Checkbox(label="Sharpen Image"),
        gr.Radio(choices=["MobileSR", "MiniSRGAN", "TinySRGAN"], label="Select Model", value="MobileSR"),
        gr.Radio(choices=["True", "False"], label="Save Output", value="False")
    ],
    outputs=gr.Image(type="pil", label="Translated Image"),
    title="Correction App",
    description="Upload an image and get the translated version. Some images may be blurry, you can tick the checkbox to sharpen them. Choose between three different models for translation.",
    allow_flagging='never'
)

# Launch the Gradio app
interface.launch()