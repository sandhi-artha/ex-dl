from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

def load_model():
    # source: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1,  # if not specified, will train from scratch
    )
    return model