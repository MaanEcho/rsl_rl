import torch
import unittest
from rsl_rl.modules.quantile_network import QuantileNetwork


class QuantileNetworkTest(unittest.TestCase):
    def test_l1_loss(self):
        qn = QuantileNetwork(10, 1, quantile_count=5)

        prediction = torch.tensor(
            [
                [0.8510, 0.2329, 0.4244, 0.5241, 0.2144],
                [0.7693, 0.2522, 0.3909, 0.0858, 0.7914],
                [0.8701, 0.2144, 0.9661, 0.9975, 0.5043],
                [0.2653, 0.6951, 0.9787, 0.2244, 0.0430],
                [0.7907, 0.5209, 0.7276, 0.1735, 0.2757],
                [0.1696, 0.7167, 0.6363, 0.2188, 0.7025],
                [0.0445, 0.6008, 0.5334, 0.1838, 0.7387],
                [0.4934, 0.5117, 0.4488, 0.0591, 0.6442],
            ]
        )
        target = torch.tensor(
            [
                [0.3918, 0.8979, 0.4347, 0.1076, 0.5303],
                [0.5449, 0.9974, 0.3197, 0.8686, 0.0631],
                [0.7397, 0.7734, 0.6559, 0.3020, 0.7229],
                [0.9519, 0.8138, 0.1502, 0.3445, 0.3356],
                [0.8970, 0.0910, 0.7536, 0.6069, 0.2556],
                [0.1741, 0.6863, 0.7142, 0.2911, 0.3142],
                [0.8835, 0.0215, 0.4774, 0.5362, 0.4998],
                [0.8037, 0.8269, 0.5518, 0.4368, 0.5323],
            ]
        )

        loss = qn.quantile_l1_loss(prediction, target)

        self.assertAlmostEqual(loss.item(), 0.16419549)

    def test_l1_loss_3d(self):
        qn = QuantileNetwork(10, 1, quantile_count=5)

        prediction = torch.tensor(
            [
                [
                    [0.8510, 0.2329, 0.4244, 0.5241, 0.2144],
                    [0.7693, 0.2522, 0.3909, 0.0858, 0.7914],
                    [0.8701, 0.2144, 0.9661, 0.9975, 0.5043],
                    [0.2653, 0.6951, 0.9787, 0.2244, 0.0430],
                    [0.7907, 0.5209, 0.7276, 0.1735, 0.2757],
                    [0.1696, 0.7167, 0.6363, 0.2188, 0.7025],
                    [0.0445, 0.6008, 0.5334, 0.1838, 0.7387],
                    [0.4934, 0.5117, 0.4488, 0.0591, 0.6442],
                ],
                [
                    [0.6874, 0.6214, 0.7796, 0.8148, 0.2070],
                    [0.0276, 0.5764, 0.5516, 0.9682, 0.6901],
                    [0.4020, 0.7084, 0.9965, 0.4311, 0.3789],
                    [0.5350, 0.9431, 0.1032, 0.6959, 0.4992],
                    [0.5059, 0.5479, 0.2302, 0.6753, 0.1593],
                    [0.6753, 0.4590, 0.9956, 0.6117, 0.1410],
                    [0.7464, 0.7184, 0.2972, 0.7694, 0.7999],
                    [0.3907, 0.2112, 0.6485, 0.0139, 0.6252],
                ],
            ]
        )
        target = torch.tensor(
            [
                [
                    [0.3918, 0.8979, 0.4347, 0.1076, 0.5303],
                    [0.5449, 0.9974, 0.3197, 0.8686, 0.0631],
                    [0.7397, 0.7734, 0.6559, 0.3020, 0.7229],
                    [0.9519, 0.8138, 0.1502, 0.3445, 0.3356],
                    [0.8970, 0.0910, 0.7536, 0.6069, 0.2556],
                    [0.1741, 0.6863, 0.7142, 0.2911, 0.3142],
                    [0.8835, 0.0215, 0.4774, 0.5362, 0.4998],
                    [0.8037, 0.8269, 0.5518, 0.4368, 0.5323],
                ],
                [
                    [0.5120, 0.7683, 0.3579, 0.8640, 0.4374],
                    [0.2533, 0.3039, 0.2214, 0.7069, 0.3093],
                    [0.6993, 0.4288, 0.0827, 0.9156, 0.2043],
                    [0.6739, 0.2303, 0.3263, 0.6884, 0.3847],
                    [0.3990, 0.1458, 0.8918, 0.8036, 0.5012],
                    [0.9061, 0.2024, 0.7276, 0.8619, 0.1198],
                    [0.7379, 0.2005, 0.7634, 0.5691, 0.6132],
                    [0.4341, 0.5711, 0.1119, 0.4286, 0.7521],
                ],
            ]
        )

        loss = qn.quantile_l1_loss(prediction, target)

        self.assertAlmostEqual(loss.item(), 0.15836075)

    def test_l1_loss_multi_output(self):
        qn = QuantileNetwork(10, 3, quantile_count=10)

        prediction = torch.tensor(
            [
                [0.3003, 0.8692, 0.4608, 0.7158, 0.2640, 0.3928, 0.4557, 0.4620, 0.1331, 0.6356],
                [0.8867, 0.1521, 0.5827, 0.0501, 0.4401, 0.7216, 0.6081, 0.5758, 0.2772, 0.6048],
                [0.0763, 0.1609, 0.1860, 0.9173, 0.2121, 0.1920, 0.8509, 0.8588, 0.3321, 0.7202],
                [0.8375, 0.5339, 0.4287, 0.9228, 0.8519, 0.0420, 0.5736, 0.9156, 0.4444, 0.2039],
                [0.0704, 0.1833, 0.0839, 0.9573, 0.9852, 0.4191, 0.3562, 0.7225, 0.8481, 0.2096],
                [0.4054, 0.8172, 0.8737, 0.2138, 0.4455, 0.7538, 0.1936, 0.9346, 0.8710, 0.0178],
                [0.2139, 0.6619, 0.6889, 0.5726, 0.0595, 0.3278, 0.7673, 0.0803, 0.0374, 0.9011],
                [0.2757, 0.0309, 0.8913, 0.0958, 0.1828, 0.9624, 0.6529, 0.7451, 0.9996, 0.8877],
                [0.0722, 0.4240, 0.0716, 0.3199, 0.5570, 0.1056, 0.5950, 0.9926, 0.2991, 0.7334],
                [0.0576, 0.6353, 0.5078, 0.4456, 0.9119, 0.6897, 0.1720, 0.5172, 0.9939, 0.5044],
                [0.6300, 0.2304, 0.4064, 0.9195, 0.3299, 0.8631, 0.5842, 0.6751, 0.2964, 0.1215],
                [0.7418, 0.5448, 0.7615, 0.6333, 0.9255, 0.1129, 0.0552, 0.4198, 0.9953, 0.7482],
                [0.9910, 0.7644, 0.7047, 0.1395, 0.3688, 0.7688, 0.8574, 0.3494, 0.6153, 0.1286],
                [0.2325, 0.7908, 0.3036, 0.4504, 0.3775, 0.6004, 0.0199, 0.9581, 0.8078, 0.8337],
                [0.4038, 0.8313, 0.5441, 0.4778, 0.5777, 0.0580, 0.5314, 0.5336, 0.0740, 0.0094],
                [0.9025, 0.5814, 0.4711, 0.2683, 0.4443, 0.5799, 0.6703, 0.2678, 0.7538, 0.1317],
                [0.6755, 0.5696, 0.3334, 0.9146, 0.6203, 0.2080, 0.0799, 0.0059, 0.8347, 0.1874],
                [0.0932, 0.0264, 0.9006, 0.3124, 0.3421, 0.8271, 0.3495, 0.2814, 0.9888, 0.5042],
                [0.4893, 0.3514, 0.2564, 0.8117, 0.3738, 0.9085, 0.3055, 0.1456, 0.3624, 0.4095],
                [0.0726, 0.2145, 0.6295, 0.7423, 0.1292, 0.7570, 0.4645, 0.0775, 0.1280, 0.7312],
                [0.8763, 0.5302, 0.8627, 0.0429, 0.2833, 0.4745, 0.6308, 0.2245, 0.2755, 0.6823],
                [0.9997, 0.3519, 0.0312, 0.1468, 0.5145, 0.0286, 0.6333, 0.1323, 0.2264, 0.9109],
                [0.7742, 0.4857, 0.0413, 0.4523, 0.6847, 0.5774, 0.9478, 0.5861, 0.9834, 0.9437],
                [0.7590, 0.5697, 0.7509, 0.3562, 0.9926, 0.3380, 0.0337, 0.7871, 0.1351, 0.9184],
                [0.5701, 0.0234, 0.8088, 0.0681, 0.7090, 0.5925, 0.5266, 0.7198, 0.4121, 0.0268],
                [0.5377, 0.1420, 0.2649, 0.0885, 0.1987, 0.1475, 0.1562, 0.2283, 0.9447, 0.4679],
                [0.0306, 0.9763, 0.1234, 0.5009, 0.8800, 0.9409, 0.3525, 0.7264, 0.2209, 0.1436],
                [0.2492, 0.4041, 0.9044, 0.3730, 0.3152, 0.7515, 0.2614, 0.9726, 0.6402, 0.5211],
                [0.8626, 0.2828, 0.6946, 0.7066, 0.4395, 0.3015, 0.2643, 0.4421, 0.6036, 0.9009],
                [0.7721, 0.1706, 0.7043, 0.4097, 0.7685, 0.3818, 0.1468, 0.6452, 0.1102, 0.1826],
                [0.7156, 0.1795, 0.5574, 0.9478, 0.0058, 0.8037, 0.8712, 0.7730, 0.5638, 0.5843],
                [0.8775, 0.6133, 0.4118, 0.3038, 0.2612, 0.2424, 0.8960, 0.8194, 0.3588, 0.3198],
            ]
        )

        target = torch.tensor(
            [
                [0.0986, 0.4029, 0.3110, 0.9976, 0.5668, 0.2658, 0.0660, 0.8492, 0.7872, 0.6368],
                [0.3556, 0.9007, 0.0227, 0.7684, 0.0105, 0.9890, 0.7468, 0.0642, 0.5164, 0.1976],
                [0.1331, 0.0998, 0.0959, 0.5596, 0.5984, 0.3880, 0.8050, 0.8320, 0.8977, 0.3486],
                [0.3297, 0.8110, 0.2844, 0.4594, 0.0739, 0.2865, 0.2957, 0.9357, 0.9898, 0.4419],
                [0.0495, 0.2826, 0.8306, 0.2968, 0.5690, 0.7251, 0.5947, 0.7526, 0.5076, 0.6480],
                [0.0381, 0.8645, 0.7774, 0.9158, 0.9682, 0.5851, 0.0913, 0.8948, 0.1251, 0.1205],
                [0.9059, 0.2758, 0.1948, 0.2694, 0.0946, 0.4381, 0.4667, 0.2176, 0.3494, 0.6073],
                [0.1778, 0.8632, 0.3015, 0.2882, 0.4214, 0.2420, 0.8394, 0.1468, 0.9679, 0.6730],
                [0.2400, 0.4344, 0.9765, 0.6544, 0.6338, 0.3434, 0.4776, 0.7981, 0.2008, 0.2267],
                [0.5574, 0.8110, 0.0264, 0.4199, 0.8178, 0.8421, 0.8237, 0.2623, 0.8025, 0.9030],
                [0.8652, 0.2872, 0.9463, 0.5543, 0.4866, 0.2842, 0.6692, 0.2306, 0.3136, 0.4570],
                [0.0651, 0.8955, 0.7531, 0.9373, 0.0265, 0.0795, 0.7755, 0.1123, 0.1920, 0.3273],
                [0.9824, 0.4177, 0.2729, 0.9447, 0.3987, 0.5495, 0.3674, 0.8067, 0.8668, 0.2394],
                [0.4874, 0.3616, 0.7577, 0.6439, 0.2927, 0.8110, 0.6821, 0.0702, 0.5514, 0.7358],
                [0.3627, 0.6392, 0.9085, 0.3646, 0.6051, 0.0586, 0.8763, 0.3899, 0.3242, 0.4598],
                [0.0167, 0.0558, 0.3862, 0.7017, 0.0403, 0.6604, 0.9992, 0.2337, 0.5128, 0.1959],
                [0.7774, 0.9201, 0.0405, 0.7894, 0.1406, 0.2458, 0.2616, 0.8787, 0.8158, 0.8591],
                [0.3225, 0.9827, 0.4032, 0.2621, 0.7949, 0.9796, 0.9480, 0.3353, 0.1430, 0.5747],
                [0.4734, 0.8714, 0.9320, 0.4265, 0.7765, 0.6980, 0.1587, 0.8784, 0.7119, 0.5141],
                [0.7263, 0.4754, 0.8234, 0.0649, 0.4343, 0.5201, 0.8274, 0.9632, 0.3525, 0.8893],
                [0.3324, 0.0142, 0.7222, 0.5026, 0.6011, 0.9275, 0.9351, 0.9236, 0.2621, 0.0768],
                [0.8456, 0.1005, 0.5550, 0.0586, 0.3811, 0.0168, 0.9724, 0.9225, 0.7242, 0.0678],
                [0.2167, 0.5423, 0.9059, 0.3320, 0.4026, 0.2128, 0.4562, 0.3564, 0.2573, 0.1076],
                [0.8385, 0.2233, 0.0736, 0.3407, 0.4702, 0.1668, 0.5174, 0.4154, 0.4407, 0.1843],
                [0.1828, 0.5321, 0.6651, 0.4108, 0.5736, 0.4012, 0.0434, 0.0034, 0.9282, 0.3111],
                [0.1754, 0.8750, 0.6629, 0.7052, 0.9739, 0.7441, 0.8954, 0.9273, 0.3836, 0.5735],
                [0.5586, 0.0381, 0.1493, 0.8575, 0.9351, 0.5222, 0.5600, 0.2369, 0.9217, 0.2545],
                [0.1054, 0.8020, 0.8463, 0.6495, 0.3011, 0.3734, 0.7263, 0.8736, 0.9258, 0.5804],
                [0.7614, 0.4748, 0.6588, 0.7717, 0.9811, 0.1659, 0.7851, 0.2135, 0.1767, 0.6724],
                [0.7655, 0.8571, 0.4224, 0.9397, 0.1363, 0.9431, 0.9326, 0.3762, 0.1077, 0.9514],
                [0.4115, 0.2169, 0.1340, 0.6564, 0.9989, 0.8068, 0.0387, 0.5064, 0.9964, 0.9427],
                [0.5760, 0.2967, 0.3891, 0.6596, 0.8037, 0.1060, 0.0102, 0.8672, 0.5922, 0.6684],
            ]
        )

        loss = qn.quantile_l1_loss(prediction, target)

        self.assertAlmostEqual(loss.item(), 0.17235948)

    def test_quantile_huber_loss(self):
        qn = QuantileNetwork(10, 1, quantile_count=5)

        prediction = torch.tensor(
            [
                [0.8510, 0.2329, 0.4244, 0.5241, 0.2144],
                [0.7693, 0.2522, 0.3909, 0.0858, 0.7914],
                [0.8701, 0.2144, 0.9661, 0.9975, 0.5043],
                [0.2653, 0.6951, 0.9787, 0.2244, 0.0430],
                [0.7907, 0.5209, 0.7276, 0.1735, 0.2757],
                [0.1696, 0.7167, 0.6363, 0.2188, 0.7025],
                [0.0445, 0.6008, 0.5334, 0.1838, 0.7387],
                [0.4934, 0.5117, 0.4488, 0.0591, 0.6442],
            ]
        )
        target = torch.tensor(
            [
                [0.3918, 0.8979, 0.4347, 0.1076, 0.5303],
                [0.5449, 0.9974, 0.3197, 0.8686, 0.0631],
                [0.7397, 0.7734, 0.6559, 0.3020, 0.7229],
                [0.9519, 0.8138, 0.1502, 0.3445, 0.3356],
                [0.8970, 0.0910, 0.7536, 0.6069, 0.2556],
                [0.1741, 0.6863, 0.7142, 0.2911, 0.3142],
                [0.8835, 0.0215, 0.4774, 0.5362, 0.4998],
                [0.8037, 0.8269, 0.5518, 0.4368, 0.5323],
            ]
        )

        loss = qn.quantile_huber_loss(prediction, target)

        self.assertAlmostEqual(loss.item(), 0.04035041)

    def test_sample_energy_loss(self):
        qn = QuantileNetwork(10, 1, quantile_count=5)

        prediction = torch.tensor(
            [
                [0.9813, 0.5331, 0.3298, 0.2428, 0.0737],
                [0.5442, 0.9623, 0.6070, 0.9360, 0.1145],
                [0.3642, 0.0887, 0.1696, 0.8027, 0.7121],
                [0.2005, 0.9889, 0.4350, 0.0301, 0.4546],
                [0.8360, 0.6766, 0.2257, 0.7589, 0.3443],
                [0.0835, 0.1747, 0.1734, 0.6668, 0.4522],
                [0.0851, 0.3146, 0.0316, 0.2250, 0.5729],
                [0.7725, 0.4596, 0.2495, 0.3633, 0.6340],
            ]
        )
        target = torch.tensor(
            [
                [0.5365, 0.1495, 0.8120, 0.2595, 0.1409],
                [0.7784, 0.7070, 0.9066, 0.0123, 0.5587],
                [0.9097, 0.0773, 0.9430, 0.2747, 0.1912],
                [0.2307, 0.5068, 0.4624, 0.6708, 0.2844],
                [0.3356, 0.5885, 0.2484, 0.8468, 0.1833],
                [0.3354, 0.8831, 0.3489, 0.7165, 0.7953],
                [0.7577, 0.8578, 0.2735, 0.1029, 0.5621],
                [0.9124, 0.3476, 0.2012, 0.5830, 0.4615],
            ]
        )

        loss = qn.sample_energy_loss(prediction, target)

        self.assertAlmostEqual(loss.item(), 0.09165202)

    def test_cvar(self):
        qn = QuantileNetwork(10, 1, quantile_count=5)
        measure = qn.measures[qn.measure_cvar](qn, 0.5)

        # Quantiles for 3 agents
        input = torch.tensor(
            [
                [0.1056, 0.0609, 0.3523, 0.3033, 0.1779],
                [0.2049, 0.1425, 0.0767, 0.1868, 0.3891],
                [0.1899, 0.1527, 0.2420, 0.2623, 0.1532],
            ]
        )
        correct_output = torch.tensor(
            [
                (0.4 * 0.0609 + 0.4 * 0.1056 + 0.2 * 0.1779),
                (0.4 * 0.0767 + 0.4 * 0.1425 + 0.2 * 0.1868),
                (0.4 * 0.1527 + 0.4 * 0.1532 + 0.2 * 0.1899),
            ]
        )

        computed_output = measure(input)

        self.assertTrue(torch.isclose(computed_output, correct_output).all())

    def test_cvar_adaptive(self):
        qn = QuantileNetwork(10, 1, quantile_count=5)

        input = torch.tensor(
            [
                [0.95, 0.21, 0.27, 0.26, 0.19],
                [0.38, 0.34, 0.18, 0.32, 0.97],
                [0.70, 0.24, 0.38, 0.89, 0.96],
            ]
        )
        confidence_levels = torch.tensor([0.1, 0.7, 0.9])
        correct_output = torch.tensor(
            [
                0.19,
                (0.18 / 3.5 + 0.32 / 3.5 + 0.34 / 3.5 + 0.38 / 7.0),
                (0.24 / 4.5 + 0.38 / 4.5 + 0.70 / 4.5 + 0.89 / 4.5 + 0.96 / 9.0),
            ]
        )

        measure = qn.measures[qn.measure_cvar](qn, confidence_levels)
        computed_output = measure(input)

        self.assertTrue(torch.isclose(computed_output, correct_output).all())
