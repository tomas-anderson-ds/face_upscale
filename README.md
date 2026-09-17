# face_upscale
Лёгкая трансформерная сеть для апскейла x2

### Использование

```sh
    python test.py --image images/image_000000197.jpg --weights checkpoint/last.pth --output result.png
```
Слева фото размером 256х256 увеличенное из картинки размером 128х128 пикселей методом BICUBIC, справа с помощью нейросети.


<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result5.png" width="512" height="256">
<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result4.png" width="512" height="256">
<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result3.png" width="512" height="256">
<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result2.png" width="512" height="256">
<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result1.png" width="512" height="256">
<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result0.png" width="512" height="256">


