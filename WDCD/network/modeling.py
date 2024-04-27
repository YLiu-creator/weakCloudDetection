from .backbone import vgg,vgg_gf1
from .backbone import vggPP,vggPP_320

def _segm_vgg(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    model = vgg.vgg16(pretrained_backbone)
    return model

def _segm_vggPP(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    model = vggPP.vgg16_PP(pretrained_backbone)
    return model

def _segm_vgg_gf1(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    model = vgg_gf1.vgg16_gf1(pretrained_backbone)
    return model

def _segm_vggPP_gf1(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    model = vggPP_320.vgg16_PP_gf1(pretrained_backbone)
    return model


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if backbone == 'vgg16':
        backbone.startswith('vgg16')
        model = _segm_vgg(arch_type, backbone, num_classes, output_stride=output_stride,
                             pretrained_backbone=pretrained_backbone)
    elif backbone == 'vgg16PP':
        backbone.startswith('vgg16PP')
        model = _segm_vggPP(arch_type, backbone, num_classes, output_stride=output_stride,
                          pretrained_backbone=pretrained_backbone)
    elif backbone == 'vgg16_gf1':
        backbone.startswith('vgg16')
        model = _segm_vgg_gf1(arch_type, backbone, num_classes, output_stride=output_stride,
                             pretrained_backbone=pretrained_backbone)
    elif backbone == 'vgg16PP_gf1':
        backbone.startswith('vgg16PP')
        model = _segm_vggPP_gf1(arch_type, backbone, num_classes, output_stride=output_stride,
                          pretrained_backbone=pretrained_backbone)

    else:
        raise NotImplementedError
    return model


def WDCD_VGG16(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a WDCD model with a VGG-16 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output strides.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('WDCD', 'vgg16', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def WDCD_VGG16PP(num_classes=21, output_stride=8, pretrained_backbone=False):
    """Constructs a WDCD model with a VGG-16 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output strides.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('WDCD', 'vgg16PP', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)

def GF1_VGG16(num_classes=2, output_stride=8, pretrained_backbone=True):
    """Constructs a WDCD model with a VGG-16 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output strides.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('WDCD', 'vgg16_gf1', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def GF1_VGG16PP(num_classes=21, output_stride=8, pretrained_backbone=False):
    """Constructs a WDCD model with a VGG-16 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output strides.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('WDCD', 'vgg16PP_gf1', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)