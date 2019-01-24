## Linear Regression

​	Linear Regression is prediction line. create one line from input and output data. Input data and Output data is represented x and y in graph.

1. Simple Linear Regression : only one x data.
2. Multi Linear Regression : input one more x data.



#### Method of least Squares

​	get *a* and *b* in *y = ax + b*.
$$
a = \frac{ \sum_{i=1}^n (x-mean(x))(y-mean(y))}{\sum_{i=1}^n(x-mean(x))^2}
$$


##### 	Cost

​		RMSE : Root Mean Squared Error
$$
RMSE = \sqrt{(mean(p_i - y_i)^2)}
$$
​		Gradient Decent : Reducing the cost

​			Learning rate : move distance on gradient decent.

```python
	tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)
```



## Logistic Regression

​	Logistic Regression create a line separating true and false. using sigmod or Relu etc.



#### Sigmoid

​	"S" shape graph.
$$
y = \frac{1}{1+e^{(ax+b)}}
$$

	##### 	Cost#####

$$
cost = -{y\log{h} + (1-y)\log{(1-h)}}
$$



## Perceptron

​	*y = ax + b* 

​	-> *y = wx + b*

​		w : weight

​		b : bias

​	**Weight sum** :
$$
b + \sum{x w} 
$$
​	**Activation function**

​		Activation function separate true and false of weighted sum's result.

​		ex) sigmoid, Relu





#### Tensorflow Function

```python
# random_uniform : 임의의 수 생성
random_uniform([1], 0, 10, datatype, seed=0)
# 0에서 10 사이에서 임의의수 1개를 뽑음
# seed=0 : 실행 시 같은 값이 나옴

#cast : Tensor 형 변환
tf.cast(tensor, dtype=type)
# tensor를 type으로 변환
```

