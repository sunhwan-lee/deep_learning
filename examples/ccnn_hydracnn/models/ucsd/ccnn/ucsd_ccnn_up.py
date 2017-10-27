from kaffe.tensorflow import Network

class UCSD_CNN(Network):
    def setup(self):
        (self.feed('data_s0')
             .conv(7, 7, 32, 1, 1, name='conv1')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(7, 7, 32, 1, 1, name='conv2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(5, 5, 64, 1, 1, name='conv3')
             .conv(1, 1, 1000, 1, 1, name='conv4')
             .conv(1, 1, 400, 1, 1, name='conv5')
             .conv(1, 1, 1, 1, 1, relu=False, name='conv6'))