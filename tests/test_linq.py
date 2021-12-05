from yutils.linq import Iteratable
import pytest

def test_get_item():
    it=Iteratable(range(10))
    for i in range(10):
        assert it[i]==i

def test_to_list():
    result = Iteratable([1,2,3,4]).to_list()
    assert isinstance(result, list)
    assert len(result)==4
    assert sum(result)==10

def test_select():
    result= Iteratable([1,2,3,4])\
            .select(lambda x: x*2)\
            .to_list()
    assert len(result)==4
    assert sum(result)==1*2+2*2+3*2+4*2

def test_where():
    result=Iteratable([1,2,3,4])\
           .select(lambda x: x*2)\
           .where(lambda x: x>2)\
           .to_list()
    assert len(result)==3
    assert sum(result)==2*2+3*2+4*2

def test_count():
    it=Iteratable(range(10))
    assert it.count() == 10
    assert it.count_if(lambda x: x%2==0) == 5

def test_first():
    it=Iteratable(range(10))
    assert it.first() == 0
    assert it.first(lambda x: x>3) == 4
    with pytest.raises(IndexError):
        Iteratable([]).first()

def test_first_or_none():
    it=Iteratable(range(10))
    assert it.first_or_none() == 0
    assert it.first_or_none(lambda x: x>3) == 4

    assert Iteratable([]).first_or_none() == None
    assert Iteratable([]).first_or_none(lambda x: x>3) == None

def test_last():
    it=Iteratable(range(10))
    assert it.last() == 9
    assert it.last(lambda x: x<7) == 6
    
    with pytest.raises(IndexError):
        Iteratable([]).last()

def test_last_or_none():
    it=Iteratable(range(10))
    assert it.last_or_none() == 9
    assert it.last_or_none(lambda x: x>3) == 9

    assert Iteratable([]).last_or_none() == None
    assert Iteratable([]).last_or_none(lambda x: x>3) == None

def test_for_each():
    result=0
    def accum_result(x):
        nonlocal result
        result += x

    Iteratable([1,2,3,4]).for_each(accum_result)
    assert result == 10

def test_take():
    result=Iteratable([1,2,3,4])\
           .take(3)\
           .to_list()
    assert len(result)==3
    assert sum(result)==1+2+3

def test_takewhile():
    result=Iteratable(range(10))\
           .takewhile(lambda x: x<5)\
           .to_list()
    assert len(result)==5
    assert sum(result)==1+2+3+4

def test_skip():    
    result=Iteratable([1,2,3,4])\
           .skip(2)\
           .to_list()
    assert len(result)==2
    assert sum(result)==3+4

def test_skipwhile():
    result=Iteratable(range(10))\
           .skipwhile(lambda x: x<5)\
           .to_list()
    assert len(result)==5
    assert sum(result)==5+6+7+8+9

@pytest.mark.skip(reason="currently broken")
def test_zip_generator():
    def my_iter():
        for i in range(10):
            yield i

    data = my_iter()
    a = Iteratable(data)

    low = a.where(lambda x: x < 5)
    high = a.where(lambda x: x >= 5)

    result = list(low.zip(high))
    print(result)

    assert result == [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
    assert list(zip(low, high))  == [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
