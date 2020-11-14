import unittest

import numpy

from CoralModel_v3.utils import SpaceTime, DataReshape


class TestSpaceTime(unittest.TestCase):

    def test_default(self):
        spacetime = SpaceTime().spacetime
        self.assertEqual(len(spacetime), 2)
        self.assertIsInstance(spacetime, tuple)

    def test_default_input(self):
        spacetime = SpaceTime(None).spacetime
        self.assertEqual(len(spacetime), 2)
        self.assertIsInstance(spacetime, tuple)

    def test_global_raise_type_error(self):
        with self.assertRaises(TypeError):
            SpaceTime(int(1))
        with self.assertRaises(TypeError):
            SpaceTime(float(1))
        with self.assertRaises(TypeError):
            SpaceTime(str(1))

    def test_global_not_raise_type_error(self):
        SpaceTime((1, 1))
        SpaceTime([1, 1])

    def test_size_error(self):
        with self.assertRaises(ValueError):
            SpaceTime((1,))
        with self.assertRaises(ValueError):
            SpaceTime((1, 1, 1))

    def test_local_raise_type_error(self):
        with self.assertRaises(TypeError):
            SpaceTime((float(1), 1))

    def test_return_type(self):
        self.assertIsInstance(SpaceTime((1, 1)).spacetime, tuple)
        self.assertIsInstance(SpaceTime([1, 1]).spacetime, tuple)


class TestDataReshape(unittest.TestCase):

    def test_default_spacetime(self):
        reshape = DataReshape()
        spacetime = SpaceTime().spacetime
        self.assertIsInstance(reshape.spacetime, tuple)
        self.assertEqual(reshape.spacetime, spacetime)

    def test_default_space(self):
        reshape = DataReshape()
        spacetime = SpaceTime().spacetime
        self.assertIsInstance(reshape.space, int)
        self.assertEqual(reshape.space, spacetime[0])

    def test_default_time(self):
        reshape = DataReshape()
        spacetime = SpaceTime().spacetime
        self.assertIsInstance(reshape.time, int)
        self.assertEqual(reshape.time, spacetime[1])

    def test_set_spacetime_raise_type_error(self):
        with self.assertRaises(TypeError):
            DataReshape(int(1))
        with self.assertRaises(TypeError):
            DataReshape(float(1))
        with self.assertRaises(TypeError):
            DataReshape(str(1))

    def test_set_spacetime_not_raise_error(self):
        DataReshape((1, 1))
        DataReshape([1, 1])

    def test_variable2array(self):
        self.assertIsInstance(DataReshape.variable2array(float(1)), numpy.ndarray)
        self.assertIsInstance(DataReshape.variable2array(int(1)), numpy.ndarray)
        with self.assertRaises(NotImplementedError):
            DataReshape.variable2array(str(1))
        self.assertIsInstance(DataReshape.variable2array((1, 1)), numpy.ndarray)
        self.assertIsInstance(DataReshape.variable2array([1, 1]), numpy.ndarray)

    def test_variable2matrix_shape_space(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4]
        matrix = reshape.variable2matrix(var, 'space')
        self.assertEqual(matrix.shape, (4, 5))

    def test_variable2matrix_shape_time(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4, 8]
        matrix = reshape.variable2matrix(var, 'time')
        self.assertEqual(matrix.shape, (4, 5))

    def test_variable2matrix_value_space(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4]
        matrix = reshape.variable2matrix(var, 'space')
        for i, row in enumerate(matrix):
            for col in row:
                self.assertEqual(col, var[i])

    def test_variable2matrix_value_time(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4, 8]
        matrix = reshape.variable2matrix(var, 'time')
        for row in matrix:
            for i, col in enumerate(row):
                self.assertEqual(col, var[i])

    def test_raise_error_space(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4, 8]
        with self.assertRaises(ValueError):
            reshape.variable2matrix(var, 'space')

    def test_raise_error_time(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4]
        with self.assertRaises(ValueError):
            reshape.variable2matrix(var, 'time')

    def test_matrix2array_space_last(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', None)
        answer = [8, 8, 8, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_space_mean(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', 'mean')
        answer = [3, 3, 3, 3]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_space_max(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', 'max')
        answer = [8, 8, 8, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_space_min(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', 'min')
        answer = [0, 0, 0, 0]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_space_sum(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', 'sum')
        answer = [15, 15, 15, 15]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_last(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', None)
        answer = [0, 1, 2, 4, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_mean(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', 'mean')
        answer = [0, 1, 2, 4, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_max(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', 'max')
        answer = [0, 1, 2, 4, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_min(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', 'min')
        answer = [0, 1, 2, 4, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_sum(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', 'sum')
        answer = [0, 4, 8, 16, 32]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)


if __name__ == '__main__':
    unittest.main()
