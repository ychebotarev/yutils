from yutils.linq import Iteratable
import pytest


def test_construct():
    with pytest.raises(TypeError):
        Iteratable(False).first()


def test_get_item():
    it = Iteratable(range(10))
    for i in range(10):
        assert it[i] == i


def test_to_list():
    result = Iteratable([1, 2, 3, 4]).to_list()
    assert isinstance(result, list)
    assert len(result) == 4
    assert sum(result) == 10


def test_select():
    result = Iteratable([1, 2, 3, 4]).select(lambda x: x * 2).to_list()
    assert len(result) == 4
    assert sum(result) == 1 * 2 + 2 * 2 + 3 * 2 + 4 * 2


def test_where():
    result = (
        Iteratable([1, 2, 3, 4])
        .select(lambda x: x * 2)
        .where(lambda x: x > 2)
        .to_list()
    )
    assert len(result) == 3
    assert sum(result) == 2 * 2 + 3 * 2 + 4 * 2


def test_count():
    it = Iteratable(range(10))
    assert it.count() == 10
    assert it.count_if(lambda x: x % 2 == 0) == 5


def test_first():
    it = Iteratable(range(10))
    assert it.first() == 0
    assert it.first(lambda x: x > 3) == 4
    with pytest.raises(IndexError):
        Iteratable([]).first()


def test_first_or_none():
    it = Iteratable(range(10))
    assert it.first_or_none() == 0
    assert it.first_or_none(lambda x: x > 3) == 4

    assert Iteratable([]).first_or_none() is None
    assert Iteratable([]).first_or_none(lambda x: x > 3) is None


def test_last():
    it = Iteratable(range(10))
    assert it.last() == 9
    assert it.last(lambda x: x < 7) == 6

    with pytest.raises(IndexError):
        Iteratable([]).last()


def test_last_or_none():
    it = Iteratable(range(10))
    assert it.last_or_none() == 9
    assert it.last_or_none(lambda x: x > 3) == 9

    assert Iteratable([]).last_or_none() is None
    assert Iteratable([]).last_or_none(lambda x: x > 3) is None


def test_for_each():
    result = 0

    def accum_result(x):
        nonlocal result
        result += x

    Iteratable([1, 2, 3, 4]).for_each(accum_result)
    assert result == 10


def test_take():
    result = Iteratable([1, 2, 3, 4]).take(3).to_list()
    assert len(result) == 3
    assert sum(result) == 1 + 2 + 3


def test_takewhile():
    result = Iteratable(range(10)).takewhile(lambda x: x < 5).to_list()
    assert len(result) == 5
    assert sum(result) == 1 + 2 + 3 + 4


def test_skip():
    result = Iteratable([1, 2, 3, 4]).skip(2).to_list()
    assert len(result) == 2
    assert sum(result) == 3 + 4


def test_skipwhile():
    result = Iteratable(range(10)).skipwhile(lambda x: x < 5).to_list()
    assert len(result) == 5
    assert sum(result) == 5 + 6 + 7 + 8 + 9


def test_zip_iterator():
    it1 = Iteratable([1, 2, 3, 4])
    it2 = Iteratable([1, 2, 3, 4])

    result = it1.zip(it2).to_list()

    print(result == [(1, 1), (2, 2), (3, 3), (4, 4)])


def text_generator():
    def my_generator():
        for i in range(10):
            yield i

    a = Iteratable(my_generator())
    low = a.where(lambda x: x < 5)

    # generator was exhausted
    high = a.where(lambda x: x >= 5)

    assert low.count() == 5
    assert high.count() == 0


def test_select_many():
    data = [[0, 1], [0, 1], [0, 1]]
    result = Iteratable(data).select_many(lambda col: col).to_list()

    assert len(result) == 6
    for i in range(6):
        assert i % 2 == result[i]


def test_chunk():
    result = Iteratable(range(10)).chunk(2).to_list()
    assert len(result) == 5
