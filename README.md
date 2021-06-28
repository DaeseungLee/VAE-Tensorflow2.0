# VAE-Tensorflow2.0

  이활석님의 [오토인코더의 모든것](https://www.youtube.com/watch?v=o_peo6U7IRM&t=2888s) 강의를 듣고 내용이 좋아서 VAE를 Tensorflow 2.0으로 구현하였다. 이론에 대한 자세한 설명은 [블로그](https://ehfkswl.tistory.com/3)에서 볼 수 있다.

![image](https://user-images.githubusercontent.com/83156421/123628125-b672f280-d84d-11eb-815a-c4618da8b36d.png)


## files
    load_data         # load data set and show img
    main              # training for model
    model             # VAE model class

## Result
- 이미지가 선명하게 나오지는 않았다. 
![image](https://user-images.githubusercontent.com/83156421/123632909-757ddc80-d853-11eb-806f-69d383eb5da5.png)


<table align='center'>
<tr align='center'>
<td> Input image </td>
<td> 2-D latent space </td>
<td> 5-D latent space </td>
<td> 10-D latent space </td>
<td> 20-D latent space </td>
</tr>
<tr>
<td><img src = 'results/input.jpg' height = '150px'>
<td><img src = 'results/dim_z_2.jpg' height = '150px'>
<td><img src = 'results/dim_z_5.jpg' height = '150px'>
<td><img src = 'results/dim_z_10.jpg' height = '150px'>
<td><img src = 'results/dim_z_20.jpg' height = '150px'>
</tr>
</table>
