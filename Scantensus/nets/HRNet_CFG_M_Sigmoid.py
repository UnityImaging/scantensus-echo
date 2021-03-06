NET_CFG = {}

NET_CFG['PRETRAINED_LAYERS'] = ['*']
NET_CFG['STEM_INPLANES'] = 64
NET_CFG['PRE_FINAL_CONV_KERNEL'] = 1
NET_CFG['FINAL_CONV_KERNEL'] = 3
NET_CFG['WITH_HEAD'] = True
NET_CFG['DOWNSAMPLE'] = 4
NET_CFG['SIGMOID'] = True

NET_CFG['STAGE1'] = {}
NET_CFG['STAGE1']['NUM_MODULES'] = 1
NET_CFG['STAGE1']['NUM_BRANCHES'] = 1
NET_CFG['STAGE1']['NUM_BLOCKS'] = [4]
NET_CFG['STAGE1']['NUM_CHANNELS'] = [64]
NET_CFG['STAGE1']['BLOCK'] = 'BOTTLENECK'
NET_CFG['STAGE1']['FUSE_METHOD'] = 'SUM'

NET_CFG['STAGE2'] = {}
NET_CFG['STAGE2']['NUM_MODULES'] = 1
NET_CFG['STAGE2']['NUM_BRANCHES'] = 2
NET_CFG['STAGE2']['NUM_BLOCKS'] = [4, 4]
NET_CFG['STAGE2']['NUM_CHANNELS'] = [18, 36]
NET_CFG['STAGE2']['BLOCK'] = 'BOTTLENECK'
NET_CFG['STAGE2']['FUSE_METHOD'] = 'SUM'

NET_CFG['STAGE3'] = {}
NET_CFG['STAGE3']['NUM_MODULES'] = 4
NET_CFG['STAGE3']['NUM_BRANCHES'] = 3
NET_CFG['STAGE3']['NUM_BLOCKS'] = [4, 4, 4]
NET_CFG['STAGE3']['NUM_CHANNELS'] = [18, 36, 72]
NET_CFG['STAGE3']['BLOCK'] = 'BOTTLENECK'
NET_CFG['STAGE3']['FUSE_METHOD'] = 'SUM'

NET_CFG['STAGE4'] = {}
NET_CFG['STAGE4']['NUM_MODULES'] = 3
NET_CFG['STAGE4']['NUM_BRANCHES'] = 4
NET_CFG['STAGE4']['NUM_BLOCKS'] = [4, 4, 4, 4]
NET_CFG['STAGE4']['NUM_CHANNELS'] = [18, 36, 72, 144]
NET_CFG['STAGE4']['BLOCK'] = 'BOTTLENECK'
NET_CFG['STAGE4']['FUSE_METHOD'] = 'SUM'

NET_CFG['DECONV'] = {}
NET_CFG['DECONV']['NUM_DECONVS'] = 1
NET_CFG['DECONV']['NUM_CHANNELS'] = [64, 64]
NET_CFG['DECONV']['NUM_BASIC_BLOCKS'] = 4
NET_CFG['DECONV']['KERNEL_SIZE'] = [2, 2]
NET_CFG['DECONV']['CAT_OUTPUT'] = [True, True]


def get_net_cfg():
    out = {}

    out['MODEL'] = {}

    out['MODEL']['EXTRA'] = NET_CFG

    return out