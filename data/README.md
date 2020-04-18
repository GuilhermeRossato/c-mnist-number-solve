# The MNIST Database of Handwritten Digits

This dataset was taken from the [this link](http://yann.lecun.com/exdb/mnist/)

It's credits goes to:

```
Yann LeCun, Courant Institute, NYU
Corinna Cortes, Google Labs, New York
Christopher J.C. Burges, Microsoft Research, Redmond
```

As is specified in the website.

The data structure of the IDX file format used by this project (as space separated bytes) is as follows:

```
0x00 0x00        (constant magic numbers)
0x0?             (Type of data, which is 08 in this case)
0x??             (The number of dimensions)
?? ?? ?? ??      (size in dimension 0 as a 32-bit MSB first/high endian)
?? ?? ?? ??      (size in dimension 1 as a 32-bit ...)
?? ?? ?? ??      (size in dimension ...)
?? ?? ...... ??? (The data in bytes, where the index in the last dimension changes the fastest)
(Obs: the size of the data is the multiplication of each dimension)
```

Don't forget that if you have little-endian system, you will have to invert the reading of the dimension size.
