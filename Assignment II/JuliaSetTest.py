import pytest
import JuliaSet


#def test_data():
#        return [(1000, 300, 33219980), (100, 300, 334236)]

@pytest.mark.parametrize("num1, num2, expected", [(1000, 300, 33219980)])
def test_sum(num1, num2, expected):
        assert JuliaSet.calc_pure_python(num1,num2) == expected


"""@pytest.fixture
def test_data():
        return [(1000, 300, 33219980), (100, 300, 334236)]
def test_sum(test_data):
        for data in test_data:
            width = data[0]
            iterations = data[1]
            expected = data[2]
            assert JuliaSet.calc_pure_python(width, iterations) == expected"""