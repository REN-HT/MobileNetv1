# MobileNetv1
This is MobileNetv1 Body Architecture

## prerequisite  
windows10(64)  
python3.7.6

## note
you can use some brief codes to verify this model as follows:  
*input size should be 224 * 224*
```
model=MobileNet()  
print(model)  
input=Variable(torch.randn(1,3,224,224))  
output=model(input)  
print(output)  
```
        



