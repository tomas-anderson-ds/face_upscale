# Face Upscale
Лёгкая и быстрая трансформерная сеть для апскейла x2

### Инференс

```sh
    python test.py --image images/image_000000197.jpg --weights checkpoint/last.pth --output result.png
```
Время инференса на изображении 128х128 -> 256х256 используя GPU RTX 5060 в среднем составляет 0.07 секунды (7 миллисекунд)

### Тренировка

```sh
    python train.py --data_dir "папка с изображениями" --lr 0.001 --w_perceptual 0.1 --w_fft 0.1 --output_dir checkpoint --batch_size 4 --num_workers 4 --dim 64 --depth 8 --heads 8 --window 8
```

Для тренировки использовались изображения из набора CelebAMask-HQ, размер изображений 1024х1024, при использовании изображений другого размера вам нужно будет изменить класс SRDataset в файле train.py.
Сеть тренировалась 90 эпох, наверно стоит потренировать 300 эпох, и поиграть с аугментацией для лучшего качества.
Если нужно больше резкости при апскейле увеличьте w_perceptual до 0.5, при этом возможны довольно сильные артефакты.

Слева фото размером 256х256 увеличенное из картинки размером 128х128 пикселей методом BICUBIC, справа с помощью нейросети.

<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result6.png" width="512" height="256">
<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result5.png" width="512" height="256">
<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result4.png" width="512" height="256">
<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result3.png" width="512" height="256">
<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result2.png" width="512" height="256">
<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result1.png" width="512" height="256">
<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result0.png" width="512" height="256">
<img src="https://github.com/tomas-anderson-ds/face_upscale/blob/main/images/result7.png" width="512" height="256">


