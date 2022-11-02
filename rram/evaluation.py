import math 

def RRAM_wlatency(xbar):
    write_latency = 1.76e-4
    # row_write_latency = 20.596e-9
    array_writelatency = write_latency * xbar  # s
    print('    array_writelatency: ', array_writelatency, ' s')
    return array_writelatency
def RRAM_wenergy(xbar):
    write_energy = 6.76e-7
    # row_write_energy = 85.929e-12
    array_writeenergy = write_energy * xbar #J each row
    print('    array_writeenergy: ', array_writeenergy, ' J')
    return array_writeenergy

def ReRAM_write_latency(row): #需要写多少次
    #如果所有阵列并行写，则time == 1
    row_write_latency = 5.12e-7
    # row_write_latency = 20.596e-9
    array_writelatency = row_write_latency * row  # s
    print('    array_writelatency: ', array_writelatency, ' s')
    return array_writelatency

def ReRAM_write_energy(row): 
    row_write_energy = 2.2e-9 
    # row_write_energy = 85.929e-12
    array_writeenergy = row_write_energy * row #J each row
    print('    array_writeenergy: ', array_writeenergy, ' J')
    return array_writeenergy

def array_energy(cycle_num, array_num): #array_num即并行打开执行计算的XB数量，cycle_num为读次数
    # rramread_energy = 1.94458e-09 #input数据为8-bit时输入一次数据的能耗，单位：J
    rramread_energy = 12.8e-9 
    array_readenergy = cycle_num * array_num * rramread_energy
    print('    array_readenergy: ', array_readenergy, ' J')
    return array_readenergy

def array_latency(cycle_num):
    # input_cycles_8 = 7.41684e-7 #intput数据为8-bit时输入一次数据的latency，单位：s
    # input_cycles = input_cycles_8
    input_cycles = 3.84e-7
    array_readtime = cycle_num * input_cycles
    print('    array_read_time: ', array_readtime, ' s')
    return array_readtime

def convlayer(configuration, xbar=128):
    "N,R,S,C,K,H,W" 
    assert(len(configuration) == 8)
    N, R, S, C, K, H, W, copys = configuration
    xbar_size = math.ceil(K /xbar) * math.ceil((R*S*C) / xbar) 
    print('required xbar_size for one layer is : ', xbar_size)
    write_energy = copys * RRAM_wenergy(xbar_size)
    write_latency = copys * RRAM_wlatency(xbar_size)

    data_iters = H*W*N 
    cycle_num = math.ceil(data_iters / copys)
    xbar_energy = array_energy(data_iters, xbar_size)
    xbar_latency = array_latency(cycle_num)

    return write_energy, write_latency, xbar_energy, xbar_latency

if __name__ == '__main__':
    N = 256 
    layer1 = [N, 7, 7, 3, 64, 112, 112, 256]
    layer2_1 = [N, 3, 3, 64, 64, 56, 56, 64]
    layer2_2 = [N, 3, 3, 64, 64, 56, 56, 64]
    layer3_1 = [N, 3, 3, 64, 64, 56, 56, 64]
    layer3_2 = [N, 3, 3, 64, 64, 56, 56, 64]
    layer4_1 = [N, 3, 3, 64, 128, 28, 28, 4]
    layer4_2 = [N, 3, 3, 128, 128, 28, 28, 4]
    layer5_1 = [N, 3, 3, 128, 128, 28, 28, 4]
    layer5_2 = [N, 3, 3, 128, 128, 28, 28, 4]
    layer6_1 = [N, 3, 3, 128, 256, 14, 14, 4]
    layer6_2 = [N, 3, 3, 256, 256, 14, 14, 4]
    layer7_1 = [N, 3, 3, 256, 256, 14, 14, 4]
    layer7_2 = [N, 3, 3, 256, 256, 14, 14, 4]
    layer8_1 = [N, 3, 3, 256, 512, 7, 7, 1]
    layer8_2 = [N, 3, 3, 512, 512, 7, 7, 1]
    layer9_1 = [N, 3, 3, 512, 512, 7, 7, 1]
    layer9_2 = [N, 3, 3, 512, 512, 7, 7, 1]

    # layer1 = [N, 7, 7, 3, 64, 112, 112, 1]
    # layer2_1 = [N, 3, 3, 64, 64, 56, 56, 1]
    # layer2_2 = [N, 3, 3, 64, 64, 56, 56, 1]
    # layer3_1 = [N, 3, 3, 64, 64, 56, 56, 1]
    # layer3_2 = [N, 3, 3, 64, 64, 56, 56, 1]
    # layer4_1 = [N, 3, 3, 64, 128, 28, 28, 1]
    # layer4_2 = [N, 3, 3, 128, 128, 28, 28, 1]
    # layer5_1 = [N, 3, 3, 128, 128, 28, 28, 1]
    # layer5_2 = [N, 3, 3, 128, 128, 28, 28, 1]
    # layer6_1 = [N, 3, 3, 128, 256, 14, 14, 1]
    # layer6_2 = [N, 3, 3, 256, 256, 14, 14, 1]
    # layer7_1 = [N, 3, 3, 256, 256, 14, 14, 1]
    # layer7_2 = [N, 3, 3, 256, 256, 14, 14, 1]
    # layer8_1 = [N, 3, 3, 256, 512, 7, 7, 1]
    # layer8_2 = [N, 3, 3, 512, 512, 7, 7, 1]
    # layer9_1 = [N, 3, 3, 512, 512, 7, 7, 1]
    # layer9_2 = [N, 3, 3, 512, 512, 7, 7, 1]

    layers = [layer1, layer2_1, layer2_2, layer3_1, layer3_2, layer4_1, layer4_2, layer5_1, layer5_2, layer6_1, layer6_2, layer7_1, layer7_2, layer8_1, layer8_2, layer9_1, layer9_2]
    total_write_energy, total_write_latency, total_xbar_energy, total_xbar_latency = 0,0,0,0
    for layer in layers:
        write_energy, write_latency, xbar_energy, xbar_latency = convlayer(layer, 1024)
        total_write_energy += write_energy
        total_xbar_energy += xbar_energy
        total_write_latency += write_latency
        total_xbar_latency = max(total_xbar_latency, xbar_latency)
    print('total write latency : ', total_write_latency)
    print('total xbar latency : ', total_xbar_latency)
    print('total write energy :', total_write_energy)
    print('total xbar energy :', total_xbar_energy)

    