<div class="alert alert-block alert-info" style="margin-top: 20px">
 <a href="http://cocl.us/pytorch_link_top"><img src = "http://cocl.us/Pytorch_top" width = 950, align = "center"></a>

<img src = "https://ibm.box.com/shared/static/ugcqz6ohbvff804xp84y4kqnvvk3bq1g.png" width = 200, align = "center">



<h1 align=center><font size = 5>Linear Regression with Multiple Outputs </font></h1> 


# Table of Contents
In this lab, we will  review how to make a prediction for Linear Regression with Multiple Output. 

<div class="alert alert-block alert-info" style="margin-top: 20px">

<li><a href="#ref2">Build Custom Modules </a></li>

<br>
<p></p>
Estimated Time Needed: <strong>15 min</strong>
</div>

<hr>

<a id="ref1"></a>
<h2 align=center>Class Linear  </h2>



```python
from torch import nn
import torch
```

Set the random seed:


```python
torch.manual_seed(1)
```




    <torch._C.Generator at 0x7ffad69623b0>



Set the random seed:


```python
class linear_regression(nn.Module):
    def __init__(self,input_size,output_size):
        super(linear_regression,self).__init__()
        self.linear=nn.Linear(input_size,output_size)
    def forward(self,x):
        yhat=self.linear(x)
        return yhat
```

create a linear regression  object, as our input and output will be two we set the parameters accordingly 


```python
model=linear_regression(1,10)
model(torch.tensor([1.0]))
```




    tensor([ 0.7926, -0.3920,  0.1714,  0.0797, -1.0143,  0.5097, -0.0608,  0.5047,
             1.0132,  0.1887], grad_fn=<AddBackward0>)



we can use the diagram to represent the model or object 

<img src = "https://ibm.box.com/shared/static/icmwnxru7nytlhnq5x486rffea9ncpk7.png" width = 600, align = "center">

we can see the parameters 


```python
list(model.parameters())
```




    [Parameter containing:
     tensor([[ 0.5153],
             [-0.4414],
             [-0.1939],
             [ 0.4694],
             [-0.9414],
             [ 0.5997],
             [-0.2057],
             [ 0.5087],
             [ 0.1390],
             [-0.1224]], requires_grad=True),
     Parameter containing:
     tensor([ 0.2774,  0.0493,  0.3652, -0.3897, -0.0729, -0.0900,  0.1449, -0.0040,
              0.8742,  0.3112], requires_grad=True)]



we can create a tensor with two rows representing one sample of data


```python
x=torch.tensor([[1.0]])
```

we can make a prediction 


```python
yhat=model(x)
yhat
```




    tensor([[ 0.7926, -0.3920,  0.1714,  0.0797, -1.0143,  0.5097, -0.0608,  0.5047,
              1.0132,  0.1887]], grad_fn=<AddmmBackward>)



each row in the following tensor represents a different sample 


```python
X=torch.tensor([[1.0],[1.0],[3.0]])
```

we can make a prediction using multiple samples 


```python
Yhat=model(X)
Yhat
```




    tensor([[ 0.7926, -0.3920,  0.1714,  0.0797, -1.0143,  0.5097, -0.0608,  0.5047,
              1.0132,  0.1887],
            [ 0.7926, -0.3920,  0.1714,  0.0797, -1.0143,  0.5097, -0.0608,  0.5047,
              1.0132,  0.1887],
            [ 1.8232, -1.2748, -0.2164,  1.0184, -2.8972,  1.7091, -0.4722,  1.5222,
              1.2912, -0.0561]], grad_fn=<AddmmBackward>)



the following figure represents the operation, where the red and blue  represents the different parameters, and the different shades of green represent  different samples.

 <img src = "https://ibm.box.com/shared/static/768cul6pj8hc93uh9ujpajihnp8xdukx.png" width = 600, align = "center">

<a href="http://cocl.us/pytorch_link_top">
    <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/Pytochtop.png" width="750" alt="IBM Product " />
</a> 

# About the Authors:  

 [Joseph Santarcangelo]( https://www.linkedin.com/in/joseph-s-50398b136/) has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
 
Other contributors: [Michelle Carey](  https://www.linkedin.com/in/michelleccarey/) 

Copyright &copy; 2018 <a href="cognitiveclass.ai?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu">cognitiveclass.ai</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.
