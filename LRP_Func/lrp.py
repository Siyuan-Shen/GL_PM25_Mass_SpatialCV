#################################################################
## This .py package is used
#################################################################


"""Class for layer-wise relevance propagation.

Layer-wise relevance propagation for VGG-like networks from PyTorch's Model Zoo.
Implementation can be adapted to work with other architectures as well by adding the corresponding operations.

    Typical usage example:

        model = torchvision.models.vgg16(pretrained=True)
        lrp_model = LRPModel(model)
        r = lrp_model.forward(x)

"""

import torch
from torch import nn
from copy import deepcopy
from .utils import layers_lookup


class LRPModel(nn.Module):
    """Class wraps PyTorch model to perform layer-wise relevance propagation."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()  # self.model.train() activates dropout / batch normalization etc.!

        # Parse network
        self.cov_layers = self._get_cov_layer_operations()
        self.fc_layers = self._get_fc_layer_operations()

        # Create LRP network
        self.lrp_convlayers, self.lrp_fclayers = self._create_lrp_model()

    def _create_lrp_model(self) -> torch.nn.ModuleList:
        """Method builds the model for layer-wise relevance propagation.

        Returns:
            LRP-model as module list.

        """
        # Clone layers from original model. This is necessary as we might modify the weights.
        conv_layers = deepcopy(self.cov_layers)
        fc_layers = deepcopy(self.fc_layers)
        lookup_table = layers_lookup()

        # Run backwards through layers
        for i, layer in enumerate(conv_layers[::-1]):
            try:
                conv_layers[i] = lookup_table[layer.__class__](layer=layer)
            except KeyError:
                message = "Layer-wise relevance propagation not implemented for " \
                          "{layer.__class__.__name__} layer."
                raise NotImplementedError(message)

        for i, layer in enumerate(fc_layers[::-1]):
            try:
                fc_layers[i] = lookup_table[layer.__class__](layer=layer)
            except KeyError:
                message = "Layer-wise relevance propagation not implemented for " \
                          "{layer.__class__.__name__} layer."
                raise NotImplementedError(message)

        return conv_layers, fc_layers

    def _get_cov_layer_operations(self) -> torch.nn.ModuleList:
        """Get all network operations and store them in a list.

        This method is adapted to VGG networks from PyTorch's Model Zoo.
        Modify this method to work also for other networks.

        Returns:
            Layers of original model stored in module list.

        """
        layers = torch.nn.ModuleList()

        for layer in self.model.conv:
            layers.append(layer)

        #layers.append(self.model.avgpool)
        #layers.append(torch.nn.Flatten(start_dim=1))


        return layers

    def _get_fc_layer_operations(self) -> torch.nn.ModuleList:
        """Get all network operations and store them in a list.

        This method is adapted to VGG networks from PyTorch's Model Zoo.
        Modify this method to work also for other networks.

        Returns:
            Layers of original model stored in module list.

        """
        layers = torch.nn.ModuleList()

        for layer in self.model.ful:  ### Example model.features output: Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), for VGG model

            layers.append(layer)

        #layers.append(self.model.avgpool)
        #layers.append(torch.nn.Flatten(start_dim=1))


        return layers

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward method that first performs standard inference followed by layer-wise relevance propagation.

        Args:
            x: Input tensor representing an image / images (N, C, H, W).

        Returns:
            Tensor holding relevance scores with dimensions (N, C, H, W).

        """
        activations = list()
        in_size = x.size(0)
        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x))
            for layer in self.cov_layers:
                if (isinstance(layer, nn.Conv2d)):
                    print('I am Conv2d!')
                    conv_in_shape_0 = x.size(0)
                    conv_in_shape_1 = x.size(1)
                    conv_in_shape_2 = x.size(2)
                    conv_in_shape_3 = x.size(3)
                    print(layer)
                    print(x.shape)
                    x = layer.forward(x)
                    activations.append(x)
                    conv_out_shape_0 = x.size(0)
                    conv_out_shape_1 = x.size(1)
                    conv_out_shape_2 = x.size(2)
                    conv_out_shape_3 = x.size(3)
                else:
                    print(layer)
                    print(x.shape)
                    x = layer.forward(x)
                    activations.append(x)
        x = x.view(in_size, -1) ## Flatten the convolutional net output
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            for layer in self.fc_layers:
                x = layer.forward(x)
                activations.append(x)

        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]
        for a in activations:
            print('This is the status of a in activations:', a.is_leaf,'\n',a.shape)

        # Initial relevance scores are the network's output activations
        relevance = activations.pop(0)
        print('Initial relevance sum: ',relevance.sum())

        # Perform relevance propagation
        for i, layer in enumerate(self.lrp_fclayers): ## Now we are running backward from the model
            if i != 1:
                temp_activation = activations.pop(0).detach()
                print('temp_activation.is_leaf?: ', temp_activation.is_leaf)
                relevance = layer.forward(temp_activation, relevance)
                print('Relevance Sum: ', i, relevance.sum())
                print(i, ' FC Layers: ',layer)


            else:
                temp_activation = activations.pop(0).view(in_size,-1).detach()
                print('temp_activation.is_leaf?: ', temp_activation.is_leaf)
                relevance = layer.forward(temp_activation, relevance)
                print('Relevance Sum: ', i, relevance.sum())
                print(i, ' FC Layers: ',layer)



        for i, layer in enumerate(self.lrp_convlayers): ## Now we are running backward from the model
            if i == 2: ## The last Convlayer (0 and 1 are activation Func and BatchNorm)
                print(i, ' Conv Layers: ', layer)

                #relevance = layer.forward(torch.Tensor(activations.pop(0).view(conv_in_shape_0,conv_in_shape_1,
                ##                                                  conv_in_shape_2,conv_in_shape_3)),
                #                          torch.Tensor(relevance.view(conv_out_shape_0,conv_out_shape_1,
                #                                        conv_out_shape_2,conv_out_shape_3)))
                temp_activation = activations.pop(0)
                
                temp_activation = temp_activation.view(conv_in_shape_0,conv_in_shape_1,
                                                                  conv_in_shape_2,conv_in_shape_3).detach()
                print('temp_activation.is_leaf?: ', temp_activation.is_leaf)                                          
                relevance = layer.forward(temp_activation,
                                          relevance.view(conv_out_shape_0,conv_out_shape_1,
                                                        conv_out_shape_2,conv_out_shape_3))
                print('Relevance Sum: ', i, relevance.sum())
                
            else:
                print(i, ' Conv Layers: ', layer)
                temp_activation = activations.pop(0).detach()
                print('temp_activation.is_leaf?: ', temp_activation.is_leaf)
                relevance = layer.forward(temp_activation, relevance)
                print('Relevance Sum: ', i, relevance.sum())
                

        return relevance.cpu().detach().numpy()

class LRPResModel(nn.Module):
    """Class wraps PyTorch model to perform layer-wise relevance propagation for Resdiual Network."""
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()  # self.model.train() activates dropout / batch normalization etc.!

        # Parse network
        self.cov_layer0, self.con_layer0_names = self._get_cov_layer_operations(self.model.layer0)
        self.cov_layer1, self.con_layer1_names = self._get_cov_layer_operations(self.model.layer1[0])
        self.cov_layer2, self.con_layer2_names = self._get_cov_layer_operations(self.model.layer2[0])
        self.cov_layer3, self.con_layer3_names = self._get_cov_layer_operations(self.model.layer3[0])
        self.cov_layer4, self.con_layer4_names = self._get_cov_layer_operations(self.model.layer4[0])

        self.fc_layers = self._get_fc_layer_operations(self.model.avgpool,self
                                                       .model.fc)
        
        # Create LRP network
        self.lrp_convlayers0 = self._create_lrp_model(self.cov_layer0,self.con_layer0_names)
        self.lrp_convlayers1 = self._create_lrp_model(self.cov_layer1,self.con_layer1_names)
        self.lrp_convlayers2 = self._create_lrp_model(self.cov_layer2,self.con_layer2_names)
        self.lrp_convlayers3 = self._create_lrp_model(self.cov_layer3,self.con_layer3_names)
        self.lrp_convlayers4 = self._create_lrp_model(self.cov_layer4,self.con_layer4_names)
        self.lrp_fclayers    = self._create_lrp_model(self.fc_layers,['avgpool','fc'])
    
    def _create_lrp_model(self,model_layers,model_names) -> torch.nn.ModuleList:
        """Method builds the model for layer-wise relevance propagation.

        Returns:
            LRP-model as module list.

        """
        # Clone layers from original model. This is necessary as we might modify the weights.
        layers = deepcopy(model_layers)
        non_downsample_layers = nn.ModuleList()
        lookup_table = layers_lookup()
        model_names = model_names[::-1]
        
        # Run backwards through layers
        for i, layer in enumerate(layers[::-1]):
            try:
                if model_names[i] != 'downsample':
                    print('Not downsample: ',layer)
                    layers[i] = lookup_table[layer.__class__](layer=layer)
                    non_downsample_layers.append(layers[i])
                else: 
                    None
            except KeyError:
                message = "Layer-wise relevance propagation not implemented for " \
                          "{layer.__class__.__name__} layer."
                raise NotImplementedError(message)
        
        return non_downsample_layers

        

    def _get_cov_layer_operations(self,model_convlayers) -> torch.nn.ModuleList:
        """Get all network operations and store them in a list.

        This method is adapted to VGG networks from PyTorch's Model Zoo.
        Modify this method to work also for other networks.

        Returns:
            Layers of original model stored in module list.

        """
        layers = torch.nn.ModuleList()
        names  = []
        for name,layer in model_convlayers.named_children():
            '''            
            if name == 'downsample':
                for subname,sublayer in layer.named_children():
                    names.append(name)
                    layers.append(sublayer)
            else:
            '''
            if (isinstance(layer, nn.Sequential)):
                for subname, sublayer in layer.named_children():
                    print(name, sublayer)
                    layers.append(sublayer)
                    names.append(name)
            else:
                print(name, layer)
                layers.append(layer)
                names.append(name)
        #layers.append(self.model.avgpool)
        #layers.append(torch.nn.Flatten(start_dim=1))
        return layers,names

    def _get_fc_layer_operations(self,*args) -> torch.nn.ModuleList:
        """Get all network operations and store them in a list.

        This method is adapted to VGG networks from PyTorch's Model Zoo.
        Modify this method to work also for other networks.

        Returns:
            Layers of original model stored in module list.

        """
        layers = torch.nn.ModuleList()
        for arg in args:
            layers.append(arg)
        #layers.append(self.model.avgpool)
        #layers.append(torch.nn.Flatten(start_dim=1))
        return layers

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward method that first performs standard inference followed by layer-wise relevance propagation.

        Args:
            x: Input tensor representing an image / images (N, C, H, W).

        Returns:
            Tensor holding relevance scores with dimensions (N, C, H, W).

        """
        activations = list()
        identity   = list()
        in_size = x.size(0)
        temp_identity_exist = False
        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x))
            print('Forward Conv Layer0')
            #### Forward Part for conv layer0
            for i,layer in enumerate(self.cov_layer0):
                initial_input = x
                
                if (isinstance(layer, nn.Conv2d)):
                    if self.con_layer0_names[i] == 'downsample':
                        temp_identity = layer.forward(initial_input)
                        temp_identity_exist = True
                    else:
                        print(layer)
                        print(x.shape)
                        x = layer.forward(x)
                        activations.append(x)
                        conv_out_shape_0 = x.size(0)
                        conv_out_shape_1 = x.size(1)
                        conv_out_shape_2 = x.size(2)
                        conv_out_shape_3 = x.size(3)
                else:
                    print(layer)
                    print(x.shape)
                    if self.con_layer0_names[i] == 'downsample':
                        temp_identity = layer.forward(temp_identity)
                        identity.append(temp_identity)
                        temp_identity_exist = True
                    elif self.con_layer0_names[i] == 'tanh2':
                        if temp_identity_exist == True:
                            x += temp_identity
                        x = layer.forward(x)
                        activations.append(x)
                    else:
                        x = layer.forward(x)
                        activations.append(x)
            
            temp_identity_exist = False
            #### Forward Part for conv layer1
            print('Forward Conv Layer1')
            for i,layer in enumerate(self.cov_layer1):
                initial_input = x
                
                if (isinstance(layer, nn.Conv2d)):
                    if self.con_layer1_names[i] == 'downsample':
                        temp_identity = layer.forward(initial_input)
                        temp_identity_exist = True
                    else:
                        print(layer)
                        print(x.shape)
                        x = layer.forward(x)
                        activations.append(x)
                        conv_out_shape_0 = x.size(0)
                        conv_out_shape_1 = x.size(1)
                        conv_out_shape_2 = x.size(2)
                        conv_out_shape_3 = x.size(3)
                else:
                    print(layer)
                    print(x.shape)
                    if self.con_layer1_names[i] == 'downsample':
                        temp_identity = layer.forward(temp_identity)
                        temp_identity_exist = True
                        identity.append(temp_identity)
                    elif self.con_layer1_names[i] == 'tanh2':
                        if temp_identity_exist == True:
                            x += temp_identity
                        x = layer.forward(x)
                        activations.append(x)
                    else:
                        x = layer.forward(x)
                        activations.append(x)
            
            #### Forward Part for conv layer2
            temp_identity_exist = False
            print('Forward Conv Layer2')
            for i,layer in enumerate(self.cov_layer2):
                initial_input = x
                
                if (isinstance(layer, nn.Conv2d)):
                    if self.con_layer2_names[i] == 'downsample':
                        temp_identity = layer.forward(initial_input)
                        temp_identity_exist =  True
                    else:
                        print(layer)
                        print(x.shape)
                        x = layer.forward(x)
                        activations.append(x)
                        conv_out_shape_0 = x.size(0)
                        conv_out_shape_1 = x.size(1)
                        conv_out_shape_2 = x.size(2)
                        conv_out_shape_3 = x.size(3)
                else:
                    print(layer)
                    print(x.shape)
                    if self.con_layer2_names[i] == 'downsample':
                        temp_identity = layer.forward(temp_identity)
                        identity.append(temp_identity)
                        temp_identity_exist = True
                    elif self.con_layer2_names[i] == 'tanh2':
                        if temp_identity_exist == True:
                            x += temp_identity
                        x = layer.forward(x)
                        activations.append(x)
                    else:
                        x = layer.forward(x)
                        activations.append(x)

        #### Forward Part for conv layer3
            temp_identity_exist = False
            print('Forward Conv Layer3')
            for i,layer in enumerate(self.cov_layer3):
                initial_input = x
                
                if (isinstance(layer, nn.Conv2d)):
                    if self.con_layer3_names[i] == 'downsample':
                        temp_identity = layer.forward(initial_input)
                        temp_identity_exist = True
                    else:
                        print(layer)
                        print(x.shape)
                        x = layer.forward(x)
                        activations.append(x)
                        conv_out_shape_0 = x.size(0)
                        conv_out_shape_1 = x.size(1)
                        conv_out_shape_2 = x.size(2)
                        conv_out_shape_3 = x.size(3)
                else:
                    print(layer)
                    print(x.shape)
                    if self.con_layer3_names[i] == 'downsample':
                        temp_identity = layer.forward(temp_identity)
                        identity.append(temp_identity)
                        temp_identity_exist = True
                    elif self.con_layer3_names[i] == 'tanh2':
                        if temp_identity_exist == True:
                            x += temp_identity
                        x = layer.forward(x)
                        activations.append(x)
                    else:
                        x = layer.forward(x)
                        activations.append(x)
                        
            #### Forward Part for conv layer4
            temp_identity_exist = False
            print('Forward Conv Layer4')
            for i,layer in enumerate(self.cov_layer4):
                initial_input = x
                
                if (isinstance(layer, nn.Conv2d)):
                    if self.con_layer4_names[i] == 'downsample':
                        temp_identity = layer.forward(initial_input)
                        temp_identity_exist = True
                    else:
                        print(layer)
                        print(x.shape)
                        x = layer.forward(x)
                        activations.append(x)
                        conv_out_shape_0 = x.size(0)
                        conv_out_shape_1 = x.size(1)
                        conv_out_shape_2 = x.size(2)
                        conv_out_shape_3 = x.size(3)
                else:
                    print(layer)
                    print(x.shape)
                    if self.con_layer4_names[i] == 'downsample':
                        temp_identity = layer.forward(temp_identity)
                        identity.append(temp_identity)
                        temp_identity_exist = True
                    elif self.con_layer4_names[i] == 'tanh2':
                        if temp_identity_exist == True:
                                x += temp_identity
                        x = layer.forward(x)
                        activations.append(x)
                    else:
                        x = layer.forward(x)
                        activations.append(x)

        
        #x = x.view(in_size, -1) ## Flatten the convolutional net output
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            for i,layer in enumerate(self.fc_layers):
                x = layer.forward(x)
                
                if i == 0:
                    x = torch.flatten(x,1)
                activations.append(x)
                
    

        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        identity = identity[::-1]

        activations = [a.data.requires_grad_(True) for a in activations]
        ###identity = [i.data.requires_grad_(True) for i in identity]
        
        #for a in activations:
        # print('This is the status of a in activations:', a.is_leaf,'\n',a.shape)
        # Initial relevance scores are the network's output activations
        relevance = activations.pop(0)
        print('Initial relevance sum: ',relevance.sum())

        # Perform relevance propagation
        for i, layer in enumerate(self.lrp_fclayers): ## Now we are running backward from the model
            if i == 0:
                print(i, ' FC Layers: ',layer)
                temp_activation = activations.pop(0).detach()#iew(in_size,-1).detach()
                print('temp_activation.is_leaf?: ', temp_activation.is_leaf)
                print('temp_activation size:', temp_activation.shape, '\n relevance size:', relevance.shape)
                relevance = layer.forward(temp_activation, relevance)
                print('Relevance Sum: ', i, relevance.sum())
            else:
                print(i, ' FC Layers: ',layer)
                temp_activation = activations.pop(0).detach()
                print('temp_activation.is_leaf?: ', temp_activation.is_leaf)
                relevance = relevance.reshape(relevance.size(0),relevance.size(1),1,1)
                print('temp_activation size:', temp_activation.shape, '\n relevance size:', relevance.shape)
                relevance = layer.forward(temp_activation, relevance)
                print('Relevance Sum: ', i, relevance.sum())
                



        for i, layer in enumerate(self.lrp_convlayers4): ## Now we are running backward from the model
            if i == 0: ## The last layer, reshape the layers (0 and 1 are activation Func and BatchNorm)
                print(i, ' Conv Layers: ', layer)
                #relevance = layer.forward(torch.Tensor(activations.pop(0).view(conv_in_shape_0,conv_in_shape_1,
                ##                                                  conv_in_shape_2,conv_in_shape_3)),
                #                          torch.Tensor(relevance.view(conv_out_shape_0,conv_out_shape_1,
                #                                        conv_out_shape_2,conv_out_shape_3)))
                temp_activation = activations.pop(0)
                
                #temp_activation = temp_activation.view(conv_in_shape_0,conv_in_shape_1,conv_in_shape_2,conv_in_shape_3).detach()
                print('temp_activation.is_leaf?: ', temp_activation.is_leaf)                                          
                relevance = layer.forward(temp_activation,
                                          relevance.view(conv_out_shape_0,conv_out_shape_1,
                                                        conv_out_shape_2,conv_out_shape_3))
                #relevance.review here is to undo the flatten
                print('Relevance Sum: ', i, relevance.sum())
            else:
                print(i, ' Conv Layers: ', layer)
                temp_activation = activations.pop(0)
                print('temp_activation.is_leaf?: ', temp_activation.is_leaf)
                relevance = layer.forward(temp_activation, relevance)
                print('Relevance Sum: ', i, relevance.sum())

        for i, layer in enumerate(self.lrp_convlayers3):
            print(i, ' Conv Layers: ', layer)
            temp_activation = activations.pop(0)
            print('temp_activation.is_leaf?: ', temp_activation.is_leaf)
            relevance = layer.forward(temp_activation, relevance)
            print('Relevance Sum: ', i, relevance.sum())
        
        for i, layer in enumerate(self.lrp_convlayers2):
            print(i, ' Conv Layers: ', layer)
            temp_activation = activations.pop(0)
            print('temp_activation.is_leaf?: ', temp_activation.is_leaf)
            relevance = layer.forward(temp_activation, relevance)
            print('Relevance Sum: ', i, relevance.sum())

        for i, layer in enumerate(self.lrp_convlayers1):
            print(i, ' Conv Layers: ', layer)
            temp_activation = activations.pop(0)
            print('temp_activation.is_leaf?: ', temp_activation.is_leaf)
            relevance = layer.forward(temp_activation, relevance)
            print('Relevance Sum: ', i, relevance.sum())  

        for i, layer in enumerate(self.lrp_convlayers0):
            print(i, ' Conv Layers: ', layer)
            temp_activation = activations.pop(0)
            print('temp_activation.is_leaf?: ', temp_activation.is_leaf)
            relevance = layer.forward(temp_activation, relevance)
            print('Relevance Sum: ', i, relevance.sum())           

        return relevance.cpu().detach().numpy()
