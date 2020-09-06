NET_CFG = {}
NET_CFG['PRETRAINED_LAYERS'] = ['*']
NET_CFG['STEM_INPLANES'] = 64
NET_CFG['FINAL_CONV_KERNEL'] = 3
NET_CFG['PRE_FINAL_CONV_KERNEL'] = 1
NET_CFG['WITH_HEAD'] = True
NET_CFG['DOWNSAMPLE'] = 4

NET_CFG['STAGE1'] = {}
NET_CFG['STAGE1']['NUM_MODULES'] = 4
NET_CFG['STAGE1']['NUM_BRANCHES'] = 1
NET_CFG['STAGE1']['NUM_BLOCKS'] = [1]
NET_CFG['STAGE1']['NUM_CHANNELS'] = [32]
NET_CFG['STAGE1']['BLOCK'] = 'BASIC'
NET_CFG['STAGE1']['FUSE_METHOD'] = 'SUM'

NET_CFG['STAGE2'] = {}
NET_CFG['STAGE2']['NUM_MODULES'] = 4
NET_CFG['STAGE2']['NUM_BRANCHES'] = 2
NET_CFG['STAGE2']['NUM_BLOCKS'] = [1, 1]
NET_CFG['STAGE2']['NUM_CHANNELS'] = [32, 64]
NET_CFG['STAGE2']['BLOCK'] = 'BASIC'
NET_CFG['STAGE2']['FUSE_METHOD'] = 'SUM'

NET_CFG['STAGE3'] = {}
NET_CFG['STAGE3']['NUM_MODULES'] = 4
NET_CFG['STAGE3']['NUM_BRANCHES'] = 3
NET_CFG['STAGE3']['NUM_BLOCKS'] = [4, 4, 4]
NET_CFG['STAGE3']['NUM_CHANNELS'] = [32, 64, 128]
NET_CFG['STAGE3']['BLOCK'] = 'BASIC'
NET_CFG['STAGE3']['FUSE_METHOD'] = 'SUM'

NET_CFG['STAGE4'] = {}
NET_CFG['STAGE4']['NUM_MODULES'] = 4
NET_CFG['STAGE4']['NUM_BRANCHES'] = 4
NET_CFG['STAGE4']['NUM_BLOCKS'] = [3, 3, 3, 3]
NET_CFG['STAGE4']['NUM_CHANNELS'] = [32, 64, 128, 256]
NET_CFG['STAGE4']['BLOCK'] = 'BASIC'
NET_CFG['STAGE4']['FUSE_METHOD'] = 'SUM'


def get_net_cfg():
    out = {}

    out['MODEL'] = {}

    out['MODEL']['EXTRA'] = NET_CFG

    return out