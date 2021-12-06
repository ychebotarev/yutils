#%%
from yutils.linq import Iteratable
result = Iteratable([1,2,3,4]).to_list()
print(result)
# %%
def my_generator():
    for i in range(10):
        yield i
a = Iteratable(my_generator())
low = a.where(lambda x: x < 5)
high = a.where(lambda x: x >= 5)
print(low.count())
print(high.count())

#%%


#%%
a = Iteratable(data)
#%%
print(a.to_list())
low = a.where(lambda x: x < 5)
high = a.where(lambda x: x >= 5)
print(low.to_list())
print(high.to_list())

result = low.zip(high)
print(result)

#%%
print(type(low._iteratable))

# %%
import inspect

print(inspect.isgenerator(data))
# %%

# %%
it1 = Iteratable([1,2,3,4])
it2 = Iteratable([1,2,3,4])

result = it1.zip(it2)

print(result.to_list() == [(1, 1), (2, 2), (3, 3), (4, 4)])

# %%
r=Iteratable([[1,2],[1,2],[1,2]]).select_many(lambda col:col).to_list()
r
# %%
