import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from Spatial_CV.utils import *

activation_func = activation_function_table()
'''
class Net(nn.Module):
    def __init__(self, nchannel):
        super(Net, self).__init__()

        self.conv = nn.Sequential(  # The first loop of ConvLay er, ReLU, Pooling
            nn.Conv2d(in_channels=nchannel,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(32),
            nn.Tanh(),


        #self.conv2 = nn.Sequential(ResidualBlocks(in_channel=32,out_channel=32,kernel_size=3,stride=1,padding=1),
        #                           nn.Tanh())

              # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        #self.conv4 = nn.Sequential(ResidualBlocks(in_channel=64,out_channel=64,kernel_size=3,stride=1,padding=1),
        #                           nn.Tanh())
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
          # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )

        # self.ful1 = nn.Sequential(nn.Linear(256  * 5 * 5, 64), nn.BatchNorm1d(64))
        self.ful = nn.Sequential(nn.Linear(256 * 5 * 5, 2),
                                 nn.Linear(2, 1)
                                 )

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv(x)
        # out = self.conv5(out)
        # out = self.conv6(out)
        out = out.view(in_size, -1)
        output = self.ful(out)

        return output
'''
'''
class ResidualBlocks(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride,padding):
        super(ResidualBlocks, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm =    nn.BatchNorm2d(out_channel)
        self.activefunc =    nn.Tanh() 
        #torch.nn.init.kaiming_normal_(self.conv.weight,
        #                              nonlinearity='relu')  # <6>
        #torch.nn.init.constant_(self.batch_norm.weight, 0.5)  # <7>
        #torch.nn.init.zeros_(self.batch_norm.bias)
    def forward(self,x):

        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activefunc(out)
        return out + x

class Net(nn.Module):
    def __init__(self, nchannel):
        super(Net, self).__init__()

        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)
        self.conv1 = nn.Sequential(  # The first loop of ConvLay er, ReLU, Pooling
            nn.Conv2d(in_channels=nchannel,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )

        self.conv2 = nn.Sequential(ResidualBlocks(in_channel=32,out_channel=32,kernel_size=3,stride=1,padding=1),
                                   nn.Tanh())

        self.conv3 = nn.Sequential(  # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )


        self.conv4 = ResidualBlocks(in_channel=64,out_channel=64,kernel_size=3,stride=1,padding=1)

        self.conv5 = nn.Sequential(  # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )

        self.conv6 = nn.Sequential(  # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )

        self.ful1 = nn.Sequential(nn.Linear(256 * 5 * 5, 64))
        self.ful2 = nn.Sequential(nn.Linear(64, 16))  # ,nn.Softmax())
        self.ful3 = nn.Sequential(nn.Linear(16, 2))
        self.ful4 = nn.Linear(2, 1)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = out.view(in_size, -1)
        out = self.ful1(out)
        out = self.ful2(out)
        out = self.ful3(out)
        output = self.ful4(out)

        return output
'''

# Standard Net 

class Net(nn.Module):
    def __init__(self, nchannel):
        super(Net, self).__init__()

        self.conv = nn.Sequential(  # The first loop of ConvLay er, ReLU, Pooling
            nn.Conv2d(in_channels=nchannel,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(32),
            nn.Tanh(),


        #self.conv2 = nn.Sequential(ResidualBlocks(in_channel=32,out_channel=32,kernel_size=3,stride=1,padding=1),
        #                           nn.Tanh())

        # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        #self.conv4 = nn.Sequential(ResidualBlocks(in_channel=64,out_channel=64,kernel_size=3,stride=1,padding=1),
        #                           nn.Tanh())

            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(128),
            nn.Tanh(),
         
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh()


        )

        # self.ful1 = nn.Sequential(nn.Linear(256  * 5 * 5, 64), nn.BatchNorm1d(64))
        self.ful = nn.Sequential(nn.Linear(256 * 5 * 5, 2),
                                 nn.Linear(2, 1))

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv(x)
        # out = self.conv5(out)
        # out = self.conv6(out)
        out = out.view(in_size, -1)
        output = self.ful(out)

        return output

'''


class ResidualBlocks(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride,padding):
        super(ResidualBlocks, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm =    nn.BatchNorm2d(out_channel)
        self.activefunc =    nn.Tanh() 
        #torch.nn.init.kaiming_normal_(self.conv.weight,
        #                              nonlinearity='relu')  # <6>
        #torch.nn.init.constant_(self.batch_norm.weight, 0.5)  # <7>
        #torch.nn.init.zeros_(self.batch_norm.bias)
    def forward(self,x):

        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activefunc(out)
        return out + x

class Net(nn.Module):
    def __init__(self, nchannel):
        super(Net, self).__init__()

        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)
        self.conv1 = nn.Sequential(  # The first loop of ConvLay er, ReLU, Pooling
            nn.Conv2d(in_channels=nchannel,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )

        self.conv2 = nn.Sequential(ResidualBlocks(in_channel=32,out_channel=32,kernel_size=3,stride=1,padding=1),
                                   nn.Tanh())

        self.conv3 = nn.Sequential(  # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )


        self.conv4 = ResidualBlocks(in_channel=64,out_channel=64,kernel_size=3,stride=1,padding=1)

        self.conv5 = nn.Sequential(  # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )

        self.conv6 = nn.Sequential(  # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )

        self.ful1 = nn.Sequential(nn.Linear(256 * 5 * 5, 64))
        self.ful2 = nn.Sequential(nn.Linear(64, 16))  # ,nn.Softmax())
        self.ful3 = nn.Sequential(nn.Linear(16, 2))
        self.ful4 = nn.Linear(2, 1)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = out.view(in_size, -1)
        out = self.ful1(out)
        out = self.ful2(out)
        out = self.ful3(out)
        output = self.ful4(out)

        return output

'''
'''
# ReLU ResNet
# Resnet 18/34使用此残差块
class BasicBlock(nn.Module):  # 卷积2层，F(X)和X的维度相等
    # expansion是F(X)相对X维度拓展的倍数
    expansion = 1  # 残差映射F(X)的维度有没有发生变化，1表示没有变化，downsample=None

    # in_channel输入特征矩阵的深度(图像通道数，如输入层有RGB三个分量，使得输入特征矩阵的深度是3)，out_channel输出特征矩阵的深度(卷积核个数)，stride卷积步长，downsample是用来将残差数据和卷积数据的shape变的相同，可以直接进行相加操作。
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN层在conv和relu层之间

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out=F(X)+X
        out += identity
        out = self.relu(out)

        return out


# Resnet 50/101/152使用此残差块
class Bottleneck(nn.Module):  # 卷积3层，F(X)和X的维度不等
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    """
    # expansion是F(X)相对X维度拓展的倍数
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
        # 此处width=out_channel

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # downsample是用来将残差数据和卷积数据的shape变的相同，可以直接进行相加操作。
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out=F(X)+X
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 nchannel, # initial input channel
                 block,  # 使用的残差块类型
                 blocks_num,  # 每个卷积层，使用残差块的个数
                 num_classes=1,  # 训练集标签的分类个数
                 include_top=True,  # 是否在残差结构后接上pooling、fc、softmax
                 groups=1,
                 width_per_group=64):

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 第一层卷积输出特征矩阵的深度，也是后面层输入特征矩阵的深度

        self.groups = groups
        self.width_per_group = width_per_group

        # 输入层有RGB三个分量，使得输入特征矩阵的深度是3
        self.conv1 = nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # _make_layer(残差块类型，残差块中第一个卷积层的卷积核个数，残差块个数，残差块中卷积步长)函数：生成多个连续的残差块的残差结构
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:  # 默认为True，接上pooling、fc、softmax
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化下采样，无论输入矩阵的shape为多少，output size均为的高宽均为1x1
            # 使矩阵展平为向量，如（W,H,C）->(1,1,W*H*C)，深度为W*H*C
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层，512 * block.expansion为输入深度，num_classes为分类类别个数

        for m in self.modules():  # 初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # _make_layer()函数：生成多个连续的残差块，(残差块类型，残差块中第一个卷积层的卷积核个数，残差块个数，残差块中卷积步长)
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        # 寻找：卷积步长不为1或深度扩张有变化，导致F(X)与X的shape不同的残差块，就要对X定义下采样函数，使之shape相同
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        # layers用于顺序储存各连续残差块
        # 每个残差结构，第一个残差块均为需要对X下采样的残差块，后面的残差块不需要对X下采样
        layers = []
        # 添加第一个残差块，第一个残差块均为需要对X下采样的残差块
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion
        # 后面的残差块不需要对X下采样
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        # 以非关键字参数形式，将layers列表，传入Sequential(),使其中残差块串联为一个残差结构
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:  # 一般为True
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


'''


# Resnet 
class BasicBlock(nn.Module):
    expansion = 1  
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  
        self.tanh1 = activation_func
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.tanh2 = activation_func
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out=F(X)+X
        out += identity
        out = self.tanh2(out)

        return out


# Resnet 50/101/152
class Bottleneck(nn.Module):  
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
        # 此处width=out_channel
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.Tanh = activation_func

    def forward(self, x):
        identity = x
       
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.Tanh(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out=F(X)+X
        out += identity
        out = self.Tanh(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 nchannel, # initial input channel
                 block,  # block types
                 blocks_num,  
                 num_classes=1,  
                 include_top=True, 
                 groups=1,
                 width_per_group=64):

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  

        self.groups = groups
        self.width_per_group = width_per_group
        self.actfunc = activation_func
        # 输入层有RGB三个分量，使得输入特征矩阵的深度是3
        
        #self.conv1 = nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.in_channel)

        #self.tanh = nn.Tanh()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer0 = nn.Sequential(self.conv1,self.bn1,self.tanh,self.maxpool)
        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=5, stride=1,padding=1, bias=False)
        ,nn.BatchNorm2d(self.in_channel)
        ,activation_func)
       # ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1)

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

   
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

    
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion # The input channel changed here!``
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            #x = self.actfunc(x)
            x = self.fc(x)

        return x



