ќ
яО
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.12.02v2.12.0-rc1-12-g0db597d0d758Я
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0

Adam/v/dense_649/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/dense_649/bias
{
)Adam/v/dense_649/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_649/bias*
_output_shapes
:*
dtype0

Adam/m/dense_649/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/dense_649/bias
{
)Adam/m/dense_649/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_649/bias*
_output_shapes
:*
dtype0

Adam/v/dense_649/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/v/dense_649/kernel

+Adam/v/dense_649/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_649/kernel*
_output_shapes
:	*
dtype0

Adam/m/dense_649/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/m/dense_649/kernel

+Adam/m/dense_649/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_649/kernel*
_output_shapes
:	*
dtype0

Adam/v/conv2d_2599/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/conv2d_2599/bias

+Adam/v/conv2d_2599/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2599/bias*
_output_shapes	
:*
dtype0

Adam/m/conv2d_2599/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/conv2d_2599/bias

+Adam/m/conv2d_2599/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2599/bias*
_output_shapes	
:*
dtype0

Adam/v/conv2d_2599/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/v/conv2d_2599/kernel

-Adam/v/conv2d_2599/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2599/kernel*'
_output_shapes
:@*
dtype0

Adam/m/conv2d_2599/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameAdam/m/conv2d_2599/kernel

-Adam/m/conv2d_2599/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2599/kernel*'
_output_shapes
:@*
dtype0

Adam/v/conv2d_2598/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/v/conv2d_2598/bias

+Adam/v/conv2d_2598/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2598/bias*
_output_shapes
:@*
dtype0

Adam/m/conv2d_2598/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/m/conv2d_2598/bias

+Adam/m/conv2d_2598/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2598/bias*
_output_shapes
:@*
dtype0

Adam/v/conv2d_2598/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdam/v/conv2d_2598/kernel

-Adam/v/conv2d_2598/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2598/kernel*&
_output_shapes
: @*
dtype0

Adam/m/conv2d_2598/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdam/m/conv2d_2598/kernel

-Adam/m/conv2d_2598/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2598/kernel*&
_output_shapes
: @*
dtype0

Adam/v/conv2d_2597/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/v/conv2d_2597/bias

+Adam/v/conv2d_2597/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2597/bias*
_output_shapes
: *
dtype0

Adam/m/conv2d_2597/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/m/conv2d_2597/bias

+Adam/m/conv2d_2597/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2597/bias*
_output_shapes
: *
dtype0

Adam/v/conv2d_2597/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/v/conv2d_2597/kernel

-Adam/v/conv2d_2597/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2597/kernel*&
_output_shapes
: *
dtype0

Adam/m/conv2d_2597/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/m/conv2d_2597/kernel

-Adam/m/conv2d_2597/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2597/kernel*&
_output_shapes
: *
dtype0

Adam/v/conv2d_2596/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/conv2d_2596/bias

+Adam/v/conv2d_2596/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2596/bias*
_output_shapes
:*
dtype0

Adam/m/conv2d_2596/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/conv2d_2596/bias

+Adam/m/conv2d_2596/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2596/bias*
_output_shapes
:*
dtype0

Adam/v/conv2d_2596/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/v/conv2d_2596/kernel

-Adam/v/conv2d_2596/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_2596/kernel*&
_output_shapes
:*
dtype0

Adam/m/conv2d_2596/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/m/conv2d_2596/kernel

-Adam/m/conv2d_2596/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_2596/kernel*&
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
t
dense_649/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_649/bias
m
"dense_649/bias/Read/ReadVariableOpReadVariableOpdense_649/bias*
_output_shapes
:*
dtype0
}
dense_649/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_namedense_649/kernel
v
$dense_649/kernel/Read/ReadVariableOpReadVariableOpdense_649/kernel*
_output_shapes
:	*
dtype0
y
conv2d_2599/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_2599/bias
r
$conv2d_2599/bias/Read/ReadVariableOpReadVariableOpconv2d_2599/bias*
_output_shapes	
:*
dtype0

conv2d_2599/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameconv2d_2599/kernel

&conv2d_2599/kernel/Read/ReadVariableOpReadVariableOpconv2d_2599/kernel*'
_output_shapes
:@*
dtype0
x
conv2d_2598/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_2598/bias
q
$conv2d_2598/bias/Read/ReadVariableOpReadVariableOpconv2d_2598/bias*
_output_shapes
:@*
dtype0

conv2d_2598/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*#
shared_nameconv2d_2598/kernel

&conv2d_2598/kernel/Read/ReadVariableOpReadVariableOpconv2d_2598/kernel*&
_output_shapes
: @*
dtype0
x
conv2d_2597/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_2597/bias
q
$conv2d_2597/bias/Read/ReadVariableOpReadVariableOpconv2d_2597/bias*
_output_shapes
: *
dtype0

conv2d_2597/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameconv2d_2597/kernel

&conv2d_2597/kernel/Read/ReadVariableOpReadVariableOpconv2d_2597/kernel*&
_output_shapes
: *
dtype0
x
conv2d_2596/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_2596/bias
q
$conv2d_2596/bias/Read/ReadVariableOpReadVariableOpconv2d_2596/bias*
_output_shapes
:*
dtype0

conv2d_2596/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv2d_2596/kernel

&conv2d_2596/kernel/Read/ReadVariableOpReadVariableOpconv2d_2596/kernel*&
_output_shapes
:*
dtype0

!serving_default_conv2d_2596_inputPlaceholder*/
_output_shapes
:џџџџџџџџџ-*
dtype0*$
shape:џџџџџџџџџ-

StatefulPartitionedCallStatefulPartitionedCall!serving_default_conv2d_2596_inputconv2d_2596/kernelconv2d_2596/biasconv2d_2597/kernelconv2d_2597/biasconv2d_2598/kernelconv2d_2598/biasconv2d_2599/kernelconv2d_2599/biasdense_649/kerneldense_649/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_42749416

NoOpNoOp
n
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*иm
valueЮmBЫm BФm

layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Ш
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias
 !_jit_compiled_convolution_op*

"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
Ѕ
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._random_generator* 
Ш
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias
 7_jit_compiled_convolution_op*

8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
Ѕ
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
D_random_generator* 
Ш
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias
 M_jit_compiled_convolution_op*

N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses* 
Ѕ
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Z_random_generator* 
Ш
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias
 c_jit_compiled_convolution_op*

d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses* 
Ѕ
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
p_random_generator* 

q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses* 

w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses* 
Ћ
}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
L
0
 1
52
63
K4
L5
a6
b7
8
9*
L
0
 1
52
63
K4
L5
a6
b7
8
9*
* 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 


_variables
_iterations
_learning_rate
_index_dict

_momentums
_velocities
_update_step_xla*

serving_default* 

0
 1*

0
 1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

 trace_0* 
b\
VARIABLE_VALUEconv2d_2596/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv2d_2596/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 

Іtrace_0* 

Їtrace_0* 
* 
* 
* 

Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

­trace_0
Ўtrace_1* 

Џtrace_0
Аtrace_1* 
* 

50
61*

50
61*
* 

Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

Жtrace_0* 

Зtrace_0* 
b\
VARIABLE_VALUEconv2d_2597/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv2d_2597/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

Нtrace_0* 

Оtrace_0* 
* 
* 
* 

Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

Фtrace_0
Хtrace_1* 

Цtrace_0
Чtrace_1* 
* 

K0
L1*

K0
L1*
* 

Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

Эtrace_0* 

Юtrace_0* 
b\
VARIABLE_VALUEconv2d_2598/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv2d_2598/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

дtrace_0* 

еtrace_0* 
* 
* 
* 

жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

лtrace_0
мtrace_1* 

нtrace_0
оtrace_1* 
* 

a0
b1*

a0
b1*
* 

пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

фtrace_0* 

хtrace_0* 
b\
VARIABLE_VALUEconv2d_2599/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv2d_2599/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

ыtrace_0* 

ьtrace_0* 
* 
* 
* 

эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 

ђtrace_0
ѓtrace_1* 

єtrace_0
ѕtrace_1* 
* 
* 
* 
* 

іnon_trainable_variables
їlayers
јmetrics
 љlayer_regularization_losses
њlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 

ћtrace_0* 

ќtrace_0* 
* 
* 
* 

§non_trainable_variables
ўlayers
џmetrics
 layer_regularization_losses
layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEdense_649/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_649/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
З
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
 20*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
T
0
1
2
3
4
5
6
7
8
9*
T
0
1
2
3
4
5
6
7
8
 9*

Ёtrace_0
Ђtrace_1
Ѓtrace_2
Єtrace_3
Ѕtrace_4
Іtrace_5
Їtrace_6
Јtrace_7
Љtrace_8
Њtrace_9* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Ћ	variables
Ќ	keras_api

­total

Ўcount*
M
Џ	variables
А	keras_api

Бtotal

Вcount
Г
_fn_kwargs*
d^
VARIABLE_VALUEAdam/m/conv2d_2596/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/conv2d_2596/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_2596/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_2596/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/conv2d_2597/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/conv2d_2597/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_2597/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_2597/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/conv2d_2598/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/conv2d_2598/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_2598/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_2598/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/conv2d_2599/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/conv2d_2599/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_2599/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_2599/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/dense_649/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_649/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_649/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_649/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

­0
Ў1*

Ћ	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Б0
В1*

Џ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
И
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_2596/kernelconv2d_2596/biasconv2d_2597/kernelconv2d_2597/biasconv2d_2598/kernelconv2d_2598/biasconv2d_2599/kernelconv2d_2599/biasdense_649/kerneldense_649/bias	iterationlearning_rateAdam/m/conv2d_2596/kernelAdam/v/conv2d_2596/kernelAdam/m/conv2d_2596/biasAdam/v/conv2d_2596/biasAdam/m/conv2d_2597/kernelAdam/v/conv2d_2597/kernelAdam/m/conv2d_2597/biasAdam/v/conv2d_2597/biasAdam/m/conv2d_2598/kernelAdam/v/conv2d_2598/kernelAdam/m/conv2d_2598/biasAdam/v/conv2d_2598/biasAdam/m/conv2d_2599/kernelAdam/v/conv2d_2599/kernelAdam/m/conv2d_2599/biasAdam/v/conv2d_2599/biasAdam/m/dense_649/kernelAdam/v/dense_649/kernelAdam/m/dense_649/biasAdam/v/dense_649/biastotal_1count_1totalcountConst*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_save_42750105
Г
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_2596/kernelconv2d_2596/biasconv2d_2597/kernelconv2d_2597/biasconv2d_2598/kernelconv2d_2598/biasconv2d_2599/kernelconv2d_2599/biasdense_649/kerneldense_649/bias	iterationlearning_rateAdam/m/conv2d_2596/kernelAdam/v/conv2d_2596/kernelAdam/m/conv2d_2596/biasAdam/v/conv2d_2596/biasAdam/m/conv2d_2597/kernelAdam/v/conv2d_2597/kernelAdam/m/conv2d_2597/biasAdam/v/conv2d_2597/biasAdam/m/conv2d_2598/kernelAdam/v/conv2d_2598/kernelAdam/m/conv2d_2598/biasAdam/v/conv2d_2598/biasAdam/m/conv2d_2599/kernelAdam/v/conv2d_2599/kernelAdam/m/conv2d_2599/biasAdam/v/conv2d_2599/biasAdam/m/dense_649/kernelAdam/v/dense_649/kernelAdam/m/dense_649/biasAdam/v/dense_649/biastotal_1count_1totalcount*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__traced_restore_42750223Е
B
о
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749150

inputs.
conv2d_2596_42749114:"
conv2d_2596_42749116:.
conv2d_2597_42749121: "
conv2d_2597_42749123: .
conv2d_2598_42749128: @"
conv2d_2598_42749130:@/
conv2d_2599_42749135:@#
conv2d_2599_42749137:	%
dense_649_42749144:	 
dense_649_42749146:
identityЂ#conv2d_2596/StatefulPartitionedCallЂ#conv2d_2597/StatefulPartitionedCallЂ#conv2d_2598/StatefulPartitionedCallЂ#conv2d_2599/StatefulPartitionedCallЂ!dense_649/StatefulPartitionedCallЂ$dropout_2596/StatefulPartitionedCallЂ$dropout_2597/StatefulPartitionedCallЂ$dropout_2598/StatefulPartitionedCallЂ$dropout_2599/StatefulPartitionedCall
#conv2d_2596/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2596_42749114conv2d_2596_42749116*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2596_layer_call_and_return_conditional_losses_42748905џ
"max_pooling2d_2596/PartitionedCallPartitionedCall,conv2d_2596/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2596_layer_call_and_return_conditional_losses_42748835
$dropout_2596/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_2596/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42748924Д
#conv2d_2597/StatefulPartitionedCallStatefulPartitionedCall-dropout_2596/StatefulPartitionedCall:output:0conv2d_2597_42749121conv2d_2597_42749123*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2597_layer_call_and_return_conditional_losses_42748937џ
"max_pooling2d_2597/PartitionedCallPartitionedCall,conv2d_2597/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2597_layer_call_and_return_conditional_losses_42748847Љ
$dropout_2597/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_2597/PartitionedCall:output:0%^dropout_2596/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42748956Д
#conv2d_2598/StatefulPartitionedCallStatefulPartitionedCall-dropout_2597/StatefulPartitionedCall:output:0conv2d_2598_42749128conv2d_2598_42749130*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2598_layer_call_and_return_conditional_losses_42748969џ
"max_pooling2d_2598/PartitionedCallPartitionedCall,conv2d_2598/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2598_layer_call_and_return_conditional_losses_42748859Љ
$dropout_2598/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_2598/PartitionedCall:output:0%^dropout_2597/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42748988Е
#conv2d_2599/StatefulPartitionedCallStatefulPartitionedCall-dropout_2598/StatefulPartitionedCall:output:0conv2d_2599_42749135conv2d_2599_42749137*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2599_layer_call_and_return_conditional_losses_42749001
"max_pooling2d_2599/PartitionedCallPartitionedCall,conv2d_2599/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2599_layer_call_and_return_conditional_losses_42748871Њ
$dropout_2599/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_2599/PartitionedCall:output:0%^dropout_2598/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749020
,global_average_pooling2d_649/PartitionedCallPartitionedCall-dropout_2599/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_global_average_pooling2d_649_layer_call_and_return_conditional_losses_42748884ѓ
flatten_649/PartitionedCallPartitionedCall5global_average_pooling2d_649/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_flatten_649_layer_call_and_return_conditional_losses_42749029
!dense_649/StatefulPartitionedCallStatefulPartitionedCall$flatten_649/PartitionedCall:output:0dense_649_42749144dense_649_42749146*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_649_layer_call_and_return_conditional_losses_42749042y
IdentityIdentity*dense_649/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp$^conv2d_2596/StatefulPartitionedCall$^conv2d_2597/StatefulPartitionedCall$^conv2d_2598/StatefulPartitionedCall$^conv2d_2599/StatefulPartitionedCall"^dense_649/StatefulPartitionedCall%^dropout_2596/StatefulPartitionedCall%^dropout_2597/StatefulPartitionedCall%^dropout_2598/StatefulPartitionedCall%^dropout_2599/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ-: : : : : : : : : : 2J
#conv2d_2596/StatefulPartitionedCall#conv2d_2596/StatefulPartitionedCall2J
#conv2d_2597/StatefulPartitionedCall#conv2d_2597/StatefulPartitionedCall2J
#conv2d_2598/StatefulPartitionedCall#conv2d_2598/StatefulPartitionedCall2J
#conv2d_2599/StatefulPartitionedCall#conv2d_2599/StatefulPartitionedCall2F
!dense_649/StatefulPartitionedCall!dense_649/StatefulPartitionedCall2L
$dropout_2596/StatefulPartitionedCall$dropout_2596/StatefulPartitionedCall2L
$dropout_2597/StatefulPartitionedCall$dropout_2597/StatefulPartitionedCall2L
$dropout_2598/StatefulPartitionedCall$dropout_2598/StatefulPartitionedCall2L
$dropout_2599/StatefulPartitionedCall$dropout_2599/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ-
 
_user_specified_nameinputs
еO
о

#__inference__wrapped_model_42748829
conv2d_2596_inputS
9sequential_649_conv2d_2596_conv2d_readvariableop_resource:H
:sequential_649_conv2d_2596_biasadd_readvariableop_resource:S
9sequential_649_conv2d_2597_conv2d_readvariableop_resource: H
:sequential_649_conv2d_2597_biasadd_readvariableop_resource: S
9sequential_649_conv2d_2598_conv2d_readvariableop_resource: @H
:sequential_649_conv2d_2598_biasadd_readvariableop_resource:@T
9sequential_649_conv2d_2599_conv2d_readvariableop_resource:@I
:sequential_649_conv2d_2599_biasadd_readvariableop_resource:	J
7sequential_649_dense_649_matmul_readvariableop_resource:	F
8sequential_649_dense_649_biasadd_readvariableop_resource:
identityЂ1sequential_649/conv2d_2596/BiasAdd/ReadVariableOpЂ0sequential_649/conv2d_2596/Conv2D/ReadVariableOpЂ1sequential_649/conv2d_2597/BiasAdd/ReadVariableOpЂ0sequential_649/conv2d_2597/Conv2D/ReadVariableOpЂ1sequential_649/conv2d_2598/BiasAdd/ReadVariableOpЂ0sequential_649/conv2d_2598/Conv2D/ReadVariableOpЂ1sequential_649/conv2d_2599/BiasAdd/ReadVariableOpЂ0sequential_649/conv2d_2599/Conv2D/ReadVariableOpЂ/sequential_649/dense_649/BiasAdd/ReadVariableOpЂ.sequential_649/dense_649/MatMul/ReadVariableOpВ
0sequential_649/conv2d_2596/Conv2D/ReadVariableOpReadVariableOp9sequential_649_conv2d_2596_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0к
!sequential_649/conv2d_2596/Conv2DConv2Dconv2d_2596_input8sequential_649/conv2d_2596/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ-*
paddingSAME*
strides
Ј
1sequential_649/conv2d_2596/BiasAdd/ReadVariableOpReadVariableOp:sequential_649_conv2d_2596_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
"sequential_649/conv2d_2596/BiasAddBiasAdd*sequential_649/conv2d_2596/Conv2D:output:09sequential_649/conv2d_2596/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ-
sequential_649/conv2d_2596/ReluRelu+sequential_649/conv2d_2596/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ-Я
)sequential_649/max_pooling2d_2596/MaxPoolMaxPool-sequential_649/conv2d_2596/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides

$sequential_649/dropout_2596/IdentityIdentity2sequential_649/max_pooling2d_2596/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџВ
0sequential_649/conv2d_2597/Conv2D/ReadVariableOpReadVariableOp9sequential_649_conv2d_2597_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0і
!sequential_649/conv2d_2597/Conv2DConv2D-sequential_649/dropout_2596/Identity:output:08sequential_649/conv2d_2597/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
Ј
1sequential_649/conv2d_2597/BiasAdd/ReadVariableOpReadVariableOp:sequential_649_conv2d_2597_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ю
"sequential_649/conv2d_2597/BiasAddBiasAdd*sequential_649/conv2d_2597/Conv2D:output:09sequential_649/conv2d_2597/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 
sequential_649/conv2d_2597/ReluRelu+sequential_649/conv2d_2597/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Я
)sequential_649/max_pooling2d_2597/MaxPoolMaxPool-sequential_649/conv2d_2597/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides

$sequential_649/dropout_2597/IdentityIdentity2sequential_649/max_pooling2d_2597/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ В
0sequential_649/conv2d_2598/Conv2D/ReadVariableOpReadVariableOp9sequential_649_conv2d_2598_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0і
!sequential_649/conv2d_2598/Conv2DConv2D-sequential_649/dropout_2597/Identity:output:08sequential_649/conv2d_2598/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
Ј
1sequential_649/conv2d_2598/BiasAdd/ReadVariableOpReadVariableOp:sequential_649_conv2d_2598_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
"sequential_649/conv2d_2598/BiasAddBiasAdd*sequential_649/conv2d_2598/Conv2D:output:09sequential_649/conv2d_2598/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
sequential_649/conv2d_2598/ReluRelu+sequential_649/conv2d_2598/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Я
)sequential_649/max_pooling2d_2598/MaxPoolMaxPool-sequential_649/conv2d_2598/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides

$sequential_649/dropout_2598/IdentityIdentity2sequential_649/max_pooling2d_2598/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Г
0sequential_649/conv2d_2599/Conv2D/ReadVariableOpReadVariableOp9sequential_649_conv2d_2599_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ї
!sequential_649/conv2d_2599/Conv2DConv2D-sequential_649/dropout_2598/Identity:output:08sequential_649/conv2d_2599/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Љ
1sequential_649/conv2d_2599/BiasAdd/ReadVariableOpReadVariableOp:sequential_649_conv2d_2599_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Я
"sequential_649/conv2d_2599/BiasAddBiasAdd*sequential_649/conv2d_2599/Conv2D:output:09sequential_649/conv2d_2599/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
sequential_649/conv2d_2599/ReluRelu+sequential_649/conv2d_2599/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџа
)sequential_649/max_pooling2d_2599/MaxPoolMaxPool-sequential_649/conv2d_2599/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides

$sequential_649/dropout_2599/IdentityIdentity2sequential_649/max_pooling2d_2599/MaxPool:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
Bsequential_649/global_average_pooling2d_649/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ч
0sequential_649/global_average_pooling2d_649/MeanMean-sequential_649/dropout_2599/Identity:output:0Ksequential_649/global_average_pooling2d_649/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџq
 sequential_649/flatten_649/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ц
"sequential_649/flatten_649/ReshapeReshape9sequential_649/global_average_pooling2d_649/Mean:output:0)sequential_649/flatten_649/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
.sequential_649/dense_649/MatMul/ReadVariableOpReadVariableOp7sequential_649_dense_649_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Р
sequential_649/dense_649/MatMulMatMul+sequential_649/flatten_649/Reshape:output:06sequential_649/dense_649/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
/sequential_649/dense_649/BiasAdd/ReadVariableOpReadVariableOp8sequential_649_dense_649_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
 sequential_649/dense_649/BiasAddBiasAdd)sequential_649/dense_649/MatMul:product:07sequential_649/dense_649/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 sequential_649/dense_649/SoftmaxSoftmax)sequential_649/dense_649/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџy
IdentityIdentity*sequential_649/dense_649/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџХ
NoOpNoOp2^sequential_649/conv2d_2596/BiasAdd/ReadVariableOp1^sequential_649/conv2d_2596/Conv2D/ReadVariableOp2^sequential_649/conv2d_2597/BiasAdd/ReadVariableOp1^sequential_649/conv2d_2597/Conv2D/ReadVariableOp2^sequential_649/conv2d_2598/BiasAdd/ReadVariableOp1^sequential_649/conv2d_2598/Conv2D/ReadVariableOp2^sequential_649/conv2d_2599/BiasAdd/ReadVariableOp1^sequential_649/conv2d_2599/Conv2D/ReadVariableOp0^sequential_649/dense_649/BiasAdd/ReadVariableOp/^sequential_649/dense_649/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ-: : : : : : : : : : 2f
1sequential_649/conv2d_2596/BiasAdd/ReadVariableOp1sequential_649/conv2d_2596/BiasAdd/ReadVariableOp2d
0sequential_649/conv2d_2596/Conv2D/ReadVariableOp0sequential_649/conv2d_2596/Conv2D/ReadVariableOp2f
1sequential_649/conv2d_2597/BiasAdd/ReadVariableOp1sequential_649/conv2d_2597/BiasAdd/ReadVariableOp2d
0sequential_649/conv2d_2597/Conv2D/ReadVariableOp0sequential_649/conv2d_2597/Conv2D/ReadVariableOp2f
1sequential_649/conv2d_2598/BiasAdd/ReadVariableOp1sequential_649/conv2d_2598/BiasAdd/ReadVariableOp2d
0sequential_649/conv2d_2598/Conv2D/ReadVariableOp0sequential_649/conv2d_2598/Conv2D/ReadVariableOp2f
1sequential_649/conv2d_2599/BiasAdd/ReadVariableOp1sequential_649/conv2d_2599/BiasAdd/ReadVariableOp2d
0sequential_649/conv2d_2599/Conv2D/ReadVariableOp0sequential_649/conv2d_2599/Conv2D/ReadVariableOp2b
/sequential_649/dense_649/BiasAdd/ReadVariableOp/sequential_649/dense_649/BiasAdd/ReadVariableOp2`
.sequential_649/dense_649/MatMul/ReadVariableOp.sequential_649/dense_649/MatMul/ReadVariableOp:b ^
/
_output_shapes
:џџџџџџџџџ-
+
_user_specified_nameconv2d_2596_input


I__inference_conv2d_2596_layer_call_and_return_conditional_losses_42748905

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ-*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ-X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ-i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ-w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ-
 
_user_specified_nameinputs
Х
Q
5__inference_max_pooling2d_2597_layer_call_fn_42749678

inputs
identityс
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2597_layer_call_and_return_conditional_losses_42748847
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ

Є
1__inference_sequential_649_layer_call_fn_42749237
conv2d_2596_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:	
	unknown_8:
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallconv2d_2596_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749214o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ-: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
/
_output_shapes
:џџџџџџџџџ-
+
_user_specified_nameconv2d_2596_input
§
h
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42749062

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
и


1__inference_sequential_649_layer_call_fn_42749466

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:	
	unknown_8:
identityЂStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749214o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ-: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ-
 
_user_specified_nameinputs
Х


&__inference_signature_wrapper_42749416
conv2d_2596_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:	
	unknown_8:
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallconv2d_2596_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_42748829o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ-: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
/
_output_shapes
:џџџџџџџџџ-
+
_user_specified_nameconv2d_2596_input

l
P__inference_max_pooling2d_2596_layer_call_and_return_conditional_losses_42749626

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
§
h
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42749710

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

h
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749824

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:џџџџџџџџџd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
й

$__inference__traced_restore_42750223
file_prefix=
#assignvariableop_conv2d_2596_kernel:1
#assignvariableop_1_conv2d_2596_bias:?
%assignvariableop_2_conv2d_2597_kernel: 1
#assignvariableop_3_conv2d_2597_bias: ?
%assignvariableop_4_conv2d_2598_kernel: @1
#assignvariableop_5_conv2d_2598_bias:@@
%assignvariableop_6_conv2d_2599_kernel:@2
#assignvariableop_7_conv2d_2599_bias:	6
#assignvariableop_8_dense_649_kernel:	/
!assignvariableop_9_dense_649_bias:'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: G
-assignvariableop_12_adam_m_conv2d_2596_kernel:G
-assignvariableop_13_adam_v_conv2d_2596_kernel:9
+assignvariableop_14_adam_m_conv2d_2596_bias:9
+assignvariableop_15_adam_v_conv2d_2596_bias:G
-assignvariableop_16_adam_m_conv2d_2597_kernel: G
-assignvariableop_17_adam_v_conv2d_2597_kernel: 9
+assignvariableop_18_adam_m_conv2d_2597_bias: 9
+assignvariableop_19_adam_v_conv2d_2597_bias: G
-assignvariableop_20_adam_m_conv2d_2598_kernel: @G
-assignvariableop_21_adam_v_conv2d_2598_kernel: @9
+assignvariableop_22_adam_m_conv2d_2598_bias:@9
+assignvariableop_23_adam_v_conv2d_2598_bias:@H
-assignvariableop_24_adam_m_conv2d_2599_kernel:@H
-assignvariableop_25_adam_v_conv2d_2599_kernel:@:
+assignvariableop_26_adam_m_conv2d_2599_bias:	:
+assignvariableop_27_adam_v_conv2d_2599_bias:	>
+assignvariableop_28_adam_m_dense_649_kernel:	>
+assignvariableop_29_adam_v_dense_649_kernel:	7
)assignvariableop_30_adam_m_dense_649_bias:7
)assignvariableop_31_adam_v_dense_649_bias:%
assignvariableop_32_total_1: %
assignvariableop_33_count_1: #
assignvariableop_34_total: #
assignvariableop_35_count: 
identity_37ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9љ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*
valueB%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHК
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B к
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Њ
_output_shapes
:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOpAssignVariableOp#assignvariableop_conv2d_2596_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_1AssignVariableOp#assignvariableop_1_conv2d_2596_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_2AssignVariableOp%assignvariableop_2_conv2d_2597_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_3AssignVariableOp#assignvariableop_3_conv2d_2597_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_4AssignVariableOp%assignvariableop_4_conv2d_2598_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv2d_2598_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_6AssignVariableOp%assignvariableop_6_conv2d_2599_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2d_2599_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_649_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_649_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_12AssignVariableOp-assignvariableop_12_adam_m_conv2d_2596_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_13AssignVariableOp-assignvariableop_13_adam_v_conv2d_2596_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_14AssignVariableOp+assignvariableop_14_adam_m_conv2d_2596_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_v_conv2d_2596_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_16AssignVariableOp-assignvariableop_16_adam_m_conv2d_2597_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_17AssignVariableOp-assignvariableop_17_adam_v_conv2d_2597_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_18AssignVariableOp+assignvariableop_18_adam_m_conv2d_2597_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_v_conv2d_2597_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_20AssignVariableOp-assignvariableop_20_adam_m_conv2d_2598_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_21AssignVariableOp-assignvariableop_21_adam_v_conv2d_2598_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_m_conv2d_2598_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_v_conv2d_2598_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_24AssignVariableOp-assignvariableop_24_adam_m_conv2d_2599_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_25AssignVariableOp-assignvariableop_25_adam_v_conv2d_2599_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_m_conv2d_2599_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_v_conv2d_2599_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_m_dense_649_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_v_dense_649_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_m_dense_649_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_v_dense_649_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_1Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_1Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_34AssignVariableOpassignvariableop_34_totalIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_35AssignVariableOpassignvariableop_35_countIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ч
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: д
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Х
Q
5__inference_max_pooling2d_2598_layer_call_fn_42749735

inputs
identityс
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2598_layer_call_and_return_conditional_losses_42748859
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н
R
%__inference__update_step_xla_42703040
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	
"
_user_specified_name
gradient


I__inference_conv2d_2596_layer_call_and_return_conditional_losses_42749616

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ-*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ-X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ-i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ-w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ-
 
_user_specified_nameinputs


I__inference_conv2d_2599_layer_call_and_return_conditional_losses_42749001

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
в
Y
%__inference__update_step_xla_42703010
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
: 
"
_user_specified_name
gradient
§
h
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42749074

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

l
P__inference_max_pooling2d_2599_layer_call_and_return_conditional_losses_42749797

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
§
h
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42749086

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ї
Ѓ
.__inference_conv2d_2598_layer_call_fn_42749719

inputs!
unknown: @
	unknown_0:@
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2598_layer_call_and_return_conditional_losses_42748969w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ў
J
.__inference_flatten_649_layer_call_fn_42749840

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_flatten_649_layer_call_and_return_conditional_losses_42749029a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї

љ
G__inference_dense_649_layer_call_and_return_conditional_losses_42749042

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
т

i
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42748956

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Х
Q
5__inference_max_pooling2d_2599_layer_call_fn_42749792

inputs
identityс
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2599_layer_call_and_return_conditional_losses_42748871
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


I__inference_conv2d_2598_layer_call_and_return_conditional_losses_42749730

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

l
P__inference_max_pooling2d_2597_layer_call_and_return_conditional_losses_42749683

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
Ѓ
.__inference_conv2d_2596_layer_call_fn_42749605

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2596_layer_call_and_return_conditional_losses_42748905w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ-: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ-
 
_user_specified_nameinputs
Л
v
Z__inference_global_average_pooling2d_649_layer_call_and_return_conditional_losses_42748884

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
§
h
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42749767

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

h
/__inference_dropout_2598_layer_call_fn_42749745

inputs
identityЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42748988w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
щ

i
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749020

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯЁ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Џ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
[
?__inference_global_average_pooling2d_649_layer_call_fn_42749829

inputs
identityб
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_global_average_pooling2d_649_layer_call_and_return_conditional_losses_42748884i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
в
Y
%__inference__update_step_xla_42703020
gradient"
variable: @*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: @: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
: @
"
_user_specified_name
gradient
щ

i
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749819

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯЁ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Џ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

l
P__inference_max_pooling2d_2597_layer_call_and_return_conditional_losses_42748847

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

l
P__inference_max_pooling2d_2599_layer_call_and_return_conditional_losses_42748871

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н
e
I__inference_flatten_649_layer_call_and_return_conditional_losses_42749846

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь
K
/__inference_dropout_2598_layer_call_fn_42749750

inputs
identityР
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42749086h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
е
Z
%__inference__update_step_xla_42703030
gradient#
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:@: *
	_noinline(:($
"
_user_specified_name
variable:Q M
'
_output_shapes
:@
"
_user_specified_name
gradient
Ї

љ
G__inference_dense_649_layer_call_and_return_conditional_losses_42749866

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь
K
/__inference_dropout_2597_layer_call_fn_42749693

inputs
identityР
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42749074h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Х
Q
5__inference_max_pooling2d_2596_layer_call_fn_42749621

inputs
identityс
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2596_layer_call_and_return_conditional_losses_42748835
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ў
M
%__inference__update_step_xla_42703005
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
Б
N
%__inference__update_step_xla_42703035
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:
"
_user_specified_name
gradient
ћ
Ѕ
.__inference_conv2d_2599_layer_call_fn_42749776

inputs"
unknown:@
	unknown_0:	
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2599_layer_call_and_return_conditional_losses_42749001x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ў
M
%__inference__update_step_xla_42703015
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
: 
"
_user_specified_name
gradient


I__inference_conv2d_2597_layer_call_and_return_conditional_losses_42749673

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а
р!
!__inference__traced_save_42750105
file_prefixC
)read_disablecopyonread_conv2d_2596_kernel:7
)read_1_disablecopyonread_conv2d_2596_bias:E
+read_2_disablecopyonread_conv2d_2597_kernel: 7
)read_3_disablecopyonread_conv2d_2597_bias: E
+read_4_disablecopyonread_conv2d_2598_kernel: @7
)read_5_disablecopyonread_conv2d_2598_bias:@F
+read_6_disablecopyonread_conv2d_2599_kernel:@8
)read_7_disablecopyonread_conv2d_2599_bias:	<
)read_8_disablecopyonread_dense_649_kernel:	5
'read_9_disablecopyonread_dense_649_bias:-
#read_10_disablecopyonread_iteration:	 1
'read_11_disablecopyonread_learning_rate: M
3read_12_disablecopyonread_adam_m_conv2d_2596_kernel:M
3read_13_disablecopyonread_adam_v_conv2d_2596_kernel:?
1read_14_disablecopyonread_adam_m_conv2d_2596_bias:?
1read_15_disablecopyonread_adam_v_conv2d_2596_bias:M
3read_16_disablecopyonread_adam_m_conv2d_2597_kernel: M
3read_17_disablecopyonread_adam_v_conv2d_2597_kernel: ?
1read_18_disablecopyonread_adam_m_conv2d_2597_bias: ?
1read_19_disablecopyonread_adam_v_conv2d_2597_bias: M
3read_20_disablecopyonread_adam_m_conv2d_2598_kernel: @M
3read_21_disablecopyonread_adam_v_conv2d_2598_kernel: @?
1read_22_disablecopyonread_adam_m_conv2d_2598_bias:@?
1read_23_disablecopyonread_adam_v_conv2d_2598_bias:@N
3read_24_disablecopyonread_adam_m_conv2d_2599_kernel:@N
3read_25_disablecopyonread_adam_v_conv2d_2599_kernel:@@
1read_26_disablecopyonread_adam_m_conv2d_2599_bias:	@
1read_27_disablecopyonread_adam_v_conv2d_2599_bias:	D
1read_28_disablecopyonread_adam_m_dense_649_kernel:	D
1read_29_disablecopyonread_adam_v_dense_649_kernel:	=
/read_30_disablecopyonread_adam_m_dense_649_bias:=
/read_31_disablecopyonread_adam_v_dense_649_bias:+
!read_32_disablecopyonread_total_1: +
!read_33_disablecopyonread_count_1: )
read_34_disablecopyonread_total: )
read_35_disablecopyonread_count: 
savev2_const
identity_73ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_30/DisableCopyOnReadЂRead_30/ReadVariableOpЂRead_31/DisableCopyOnReadЂRead_31/ReadVariableOpЂRead_32/DisableCopyOnReadЂRead_32/ReadVariableOpЂRead_33/DisableCopyOnReadЂRead_33/ReadVariableOpЂRead_34/DisableCopyOnReadЂRead_34/ReadVariableOpЂRead_35/DisableCopyOnReadЂRead_35/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: {
Read/DisableCopyOnReadDisableCopyOnRead)read_disablecopyonread_conv2d_2596_kernel"/device:CPU:0*
_output_shapes
 ­
Read/ReadVariableOpReadVariableOp)read_disablecopyonread_conv2d_2596_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:}
Read_1/DisableCopyOnReadDisableCopyOnRead)read_1_disablecopyonread_conv2d_2596_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_1/ReadVariableOpReadVariableOp)read_1_disablecopyonread_conv2d_2596_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_2/DisableCopyOnReadDisableCopyOnRead+read_2_disablecopyonread_conv2d_2597_kernel"/device:CPU:0*
_output_shapes
 Г
Read_2/ReadVariableOpReadVariableOp+read_2_disablecopyonread_conv2d_2597_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: }
Read_3/DisableCopyOnReadDisableCopyOnRead)read_3_disablecopyonread_conv2d_2597_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_3/ReadVariableOpReadVariableOp)read_3_disablecopyonread_conv2d_2597_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_4/DisableCopyOnReadDisableCopyOnRead+read_4_disablecopyonread_conv2d_2598_kernel"/device:CPU:0*
_output_shapes
 Г
Read_4/ReadVariableOpReadVariableOp+read_4_disablecopyonread_conv2d_2598_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
: @}
Read_5/DisableCopyOnReadDisableCopyOnRead)read_5_disablecopyonread_conv2d_2598_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_5/ReadVariableOpReadVariableOp)read_5_disablecopyonread_conv2d_2598_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_6/DisableCopyOnReadDisableCopyOnRead+read_6_disablecopyonread_conv2d_2599_kernel"/device:CPU:0*
_output_shapes
 Д
Read_6/ReadVariableOpReadVariableOp+read_6_disablecopyonread_conv2d_2599_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@*
dtype0w
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@n
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*'
_output_shapes
:@}
Read_7/DisableCopyOnReadDisableCopyOnRead)read_7_disablecopyonread_conv2d_2599_bias"/device:CPU:0*
_output_shapes
 І
Read_7/ReadVariableOpReadVariableOp)read_7_disablecopyonread_conv2d_2599_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:}
Read_8/DisableCopyOnReadDisableCopyOnRead)read_8_disablecopyonread_dense_649_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_8/ReadVariableOpReadVariableOp)read_8_disablecopyonread_dense_649_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	{
Read_9/DisableCopyOnReadDisableCopyOnRead'read_9_disablecopyonread_dense_649_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_9/ReadVariableOpReadVariableOp'read_9_disablecopyonread_dense_649_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_iteration^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 Ё
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_learning_rate^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_12/DisableCopyOnReadDisableCopyOnRead3read_12_disablecopyonread_adam_m_conv2d_2596_kernel"/device:CPU:0*
_output_shapes
 Н
Read_12/ReadVariableOpReadVariableOp3read_12_disablecopyonread_adam_m_conv2d_2596_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_13/DisableCopyOnReadDisableCopyOnRead3read_13_disablecopyonread_adam_v_conv2d_2596_kernel"/device:CPU:0*
_output_shapes
 Н
Read_13/ReadVariableOpReadVariableOp3read_13_disablecopyonread_adam_v_conv2d_2596_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_14/DisableCopyOnReadDisableCopyOnRead1read_14_disablecopyonread_adam_m_conv2d_2596_bias"/device:CPU:0*
_output_shapes
 Џ
Read_14/ReadVariableOpReadVariableOp1read_14_disablecopyonread_adam_m_conv2d_2596_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_15/DisableCopyOnReadDisableCopyOnRead1read_15_disablecopyonread_adam_v_conv2d_2596_bias"/device:CPU:0*
_output_shapes
 Џ
Read_15/ReadVariableOpReadVariableOp1read_15_disablecopyonread_adam_v_conv2d_2596_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_16/DisableCopyOnReadDisableCopyOnRead3read_16_disablecopyonread_adam_m_conv2d_2597_kernel"/device:CPU:0*
_output_shapes
 Н
Read_16/ReadVariableOpReadVariableOp3read_16_disablecopyonread_adam_m_conv2d_2597_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_17/DisableCopyOnReadDisableCopyOnRead3read_17_disablecopyonread_adam_v_conv2d_2597_kernel"/device:CPU:0*
_output_shapes
 Н
Read_17/ReadVariableOpReadVariableOp3read_17_disablecopyonread_adam_v_conv2d_2597_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_18/DisableCopyOnReadDisableCopyOnRead1read_18_disablecopyonread_adam_m_conv2d_2597_bias"/device:CPU:0*
_output_shapes
 Џ
Read_18/ReadVariableOpReadVariableOp1read_18_disablecopyonread_adam_m_conv2d_2597_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_19/DisableCopyOnReadDisableCopyOnRead1read_19_disablecopyonread_adam_v_conv2d_2597_bias"/device:CPU:0*
_output_shapes
 Џ
Read_19/ReadVariableOpReadVariableOp1read_19_disablecopyonread_adam_v_conv2d_2597_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_20/DisableCopyOnReadDisableCopyOnRead3read_20_disablecopyonread_adam_m_conv2d_2598_kernel"/device:CPU:0*
_output_shapes
 Н
Read_20/ReadVariableOpReadVariableOp3read_20_disablecopyonread_adam_m_conv2d_2598_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
: @
Read_21/DisableCopyOnReadDisableCopyOnRead3read_21_disablecopyonread_adam_v_conv2d_2598_kernel"/device:CPU:0*
_output_shapes
 Н
Read_21/ReadVariableOpReadVariableOp3read_21_disablecopyonread_adam_v_conv2d_2598_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*&
_output_shapes
: @
Read_22/DisableCopyOnReadDisableCopyOnRead1read_22_disablecopyonread_adam_m_conv2d_2598_bias"/device:CPU:0*
_output_shapes
 Џ
Read_22/ReadVariableOpReadVariableOp1read_22_disablecopyonread_adam_m_conv2d_2598_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_23/DisableCopyOnReadDisableCopyOnRead1read_23_disablecopyonread_adam_v_conv2d_2598_bias"/device:CPU:0*
_output_shapes
 Џ
Read_23/ReadVariableOpReadVariableOp1read_23_disablecopyonread_adam_v_conv2d_2598_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_24/DisableCopyOnReadDisableCopyOnRead3read_24_disablecopyonread_adam_m_conv2d_2599_kernel"/device:CPU:0*
_output_shapes
 О
Read_24/ReadVariableOpReadVariableOp3read_24_disablecopyonread_adam_m_conv2d_2599_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@*
dtype0x
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@n
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*'
_output_shapes
:@
Read_25/DisableCopyOnReadDisableCopyOnRead3read_25_disablecopyonread_adam_v_conv2d_2599_kernel"/device:CPU:0*
_output_shapes
 О
Read_25/ReadVariableOpReadVariableOp3read_25_disablecopyonread_adam_v_conv2d_2599_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@*
dtype0x
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@n
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*'
_output_shapes
:@
Read_26/DisableCopyOnReadDisableCopyOnRead1read_26_disablecopyonread_adam_m_conv2d_2599_bias"/device:CPU:0*
_output_shapes
 А
Read_26/ReadVariableOpReadVariableOp1read_26_disablecopyonread_adam_m_conv2d_2599_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_27/DisableCopyOnReadDisableCopyOnRead1read_27_disablecopyonread_adam_v_conv2d_2599_bias"/device:CPU:0*
_output_shapes
 А
Read_27/ReadVariableOpReadVariableOp1read_27_disablecopyonread_adam_v_conv2d_2599_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_28/DisableCopyOnReadDisableCopyOnRead1read_28_disablecopyonread_adam_m_dense_649_kernel"/device:CPU:0*
_output_shapes
 Д
Read_28/ReadVariableOpReadVariableOp1read_28_disablecopyonread_adam_m_dense_649_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0p
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_29/DisableCopyOnReadDisableCopyOnRead1read_29_disablecopyonread_adam_v_dense_649_kernel"/device:CPU:0*
_output_shapes
 Д
Read_29/ReadVariableOpReadVariableOp1read_29_disablecopyonread_adam_v_dense_649_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0p
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:	
Read_30/DisableCopyOnReadDisableCopyOnRead/read_30_disablecopyonread_adam_m_dense_649_bias"/device:CPU:0*
_output_shapes
 ­
Read_30/ReadVariableOpReadVariableOp/read_30_disablecopyonread_adam_m_dense_649_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_31/DisableCopyOnReadDisableCopyOnRead/read_31_disablecopyonread_adam_v_dense_649_bias"/device:CPU:0*
_output_shapes
 ­
Read_31/ReadVariableOpReadVariableOp/read_31_disablecopyonread_adam_v_dense_649_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_32/DisableCopyOnReadDisableCopyOnRead!read_32_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_32/ReadVariableOpReadVariableOp!read_32_disablecopyonread_total_1^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_33/DisableCopyOnReadDisableCopyOnRead!read_33_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_33/ReadVariableOpReadVariableOp!read_33_disablecopyonread_count_1^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_34/DisableCopyOnReadDisableCopyOnReadread_34_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_34/ReadVariableOpReadVariableOpread_34_disablecopyonread_total^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_35/DisableCopyOnReadDisableCopyOnReadread_35_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_35/ReadVariableOpReadVariableOpread_35_disablecopyonread_count^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: і
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*
valueB%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЗ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *3
dtypes)
'2%	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_72Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_73IdentityIdentity_72:output:0^NoOp*
T0*
_output_shapes
: З
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_73Identity_73:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:%

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ў
M
%__inference__update_step_xla_42703025
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
Л;
Т
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749214

inputs.
conv2d_2596_42749178:"
conv2d_2596_42749180:.
conv2d_2597_42749185: "
conv2d_2597_42749187: .
conv2d_2598_42749192: @"
conv2d_2598_42749194:@/
conv2d_2599_42749199:@#
conv2d_2599_42749201:	%
dense_649_42749208:	 
dense_649_42749210:
identityЂ#conv2d_2596/StatefulPartitionedCallЂ#conv2d_2597/StatefulPartitionedCallЂ#conv2d_2598/StatefulPartitionedCallЂ#conv2d_2599/StatefulPartitionedCallЂ!dense_649/StatefulPartitionedCall
#conv2d_2596/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2596_42749178conv2d_2596_42749180*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2596_layer_call_and_return_conditional_losses_42748905џ
"max_pooling2d_2596/PartitionedCallPartitionedCall,conv2d_2596/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2596_layer_call_and_return_conditional_losses_42748835ђ
dropout_2596/PartitionedCallPartitionedCall+max_pooling2d_2596/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42749062Ќ
#conv2d_2597/StatefulPartitionedCallStatefulPartitionedCall%dropout_2596/PartitionedCall:output:0conv2d_2597_42749185conv2d_2597_42749187*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2597_layer_call_and_return_conditional_losses_42748937џ
"max_pooling2d_2597/PartitionedCallPartitionedCall,conv2d_2597/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2597_layer_call_and_return_conditional_losses_42748847ђ
dropout_2597/PartitionedCallPartitionedCall+max_pooling2d_2597/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42749074Ќ
#conv2d_2598/StatefulPartitionedCallStatefulPartitionedCall%dropout_2597/PartitionedCall:output:0conv2d_2598_42749192conv2d_2598_42749194*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2598_layer_call_and_return_conditional_losses_42748969џ
"max_pooling2d_2598/PartitionedCallPartitionedCall,conv2d_2598/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2598_layer_call_and_return_conditional_losses_42748859ђ
dropout_2598/PartitionedCallPartitionedCall+max_pooling2d_2598/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42749086­
#conv2d_2599/StatefulPartitionedCallStatefulPartitionedCall%dropout_2598/PartitionedCall:output:0conv2d_2599_42749199conv2d_2599_42749201*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2599_layer_call_and_return_conditional_losses_42749001
"max_pooling2d_2599/PartitionedCallPartitionedCall,conv2d_2599/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2599_layer_call_and_return_conditional_losses_42748871ѓ
dropout_2599/PartitionedCallPartitionedCall+max_pooling2d_2599/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749098
,global_average_pooling2d_649/PartitionedCallPartitionedCall%dropout_2599/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_global_average_pooling2d_649_layer_call_and_return_conditional_losses_42748884ѓ
flatten_649/PartitionedCallPartitionedCall5global_average_pooling2d_649/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_flatten_649_layer_call_and_return_conditional_losses_42749029
!dense_649/StatefulPartitionedCallStatefulPartitionedCall$flatten_649/PartitionedCall:output:0dense_649_42749208dense_649_42749210*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_649_layer_call_and_return_conditional_losses_42749042y
IdentityIdentity*dense_649/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp$^conv2d_2596/StatefulPartitionedCall$^conv2d_2597/StatefulPartitionedCall$^conv2d_2598/StatefulPartitionedCall$^conv2d_2599/StatefulPartitionedCall"^dense_649/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ-: : : : : : : : : : 2J
#conv2d_2596/StatefulPartitionedCall#conv2d_2596/StatefulPartitionedCall2J
#conv2d_2597/StatefulPartitionedCall#conv2d_2597/StatefulPartitionedCall2J
#conv2d_2598/StatefulPartitionedCall#conv2d_2598/StatefulPartitionedCall2J
#conv2d_2599/StatefulPartitionedCall#conv2d_2599/StatefulPartitionedCall2F
!dense_649/StatefulPartitionedCall!dense_649/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ-
 
_user_specified_nameinputs

h
/__inference_dropout_2596_layer_call_fn_42749631

inputs
identityЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42748924w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ
h
/__inference_dropout_2599_layer_call_fn_42749802

inputs
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749020x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю

,__inference_dense_649_layer_call_fn_42749855

inputs
unknown:	
	unknown_0:
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_649_layer_call_and_return_conditional_losses_42749042o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
т

i
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42748924

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџi
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ

Є
1__inference_sequential_649_layer_call_fn_42749173
conv2d_2596_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:	
	unknown_8:
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallconv2d_2596_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749150o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ-: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
/
_output_shapes
:џџџџџџџџџ-
+
_user_specified_nameconv2d_2596_input
Ь
K
/__inference_dropout_2596_layer_call_fn_42749636

inputs
identityР
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42749062h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
т

i
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42748988

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


I__inference_conv2d_2597_layer_call_and_return_conditional_losses_42748937

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
и


1__inference_sequential_649_layer_call_fn_42749441

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:	
	unknown_8:
identityЂStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749150o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ-: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ-
 
_user_specified_nameinputs
м;
Э
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749108
conv2d_2596_input.
conv2d_2596_42749052:"
conv2d_2596_42749054:.
conv2d_2597_42749064: "
conv2d_2597_42749066: .
conv2d_2598_42749076: @"
conv2d_2598_42749078:@/
conv2d_2599_42749088:@#
conv2d_2599_42749090:	%
dense_649_42749102:	 
dense_649_42749104:
identityЂ#conv2d_2596/StatefulPartitionedCallЂ#conv2d_2597/StatefulPartitionedCallЂ#conv2d_2598/StatefulPartitionedCallЂ#conv2d_2599/StatefulPartitionedCallЂ!dense_649/StatefulPartitionedCall
#conv2d_2596/StatefulPartitionedCallStatefulPartitionedCallconv2d_2596_inputconv2d_2596_42749052conv2d_2596_42749054*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2596_layer_call_and_return_conditional_losses_42748905џ
"max_pooling2d_2596/PartitionedCallPartitionedCall,conv2d_2596/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2596_layer_call_and_return_conditional_losses_42748835ђ
dropout_2596/PartitionedCallPartitionedCall+max_pooling2d_2596/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42749062Ќ
#conv2d_2597/StatefulPartitionedCallStatefulPartitionedCall%dropout_2596/PartitionedCall:output:0conv2d_2597_42749064conv2d_2597_42749066*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2597_layer_call_and_return_conditional_losses_42748937џ
"max_pooling2d_2597/PartitionedCallPartitionedCall,conv2d_2597/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2597_layer_call_and_return_conditional_losses_42748847ђ
dropout_2597/PartitionedCallPartitionedCall+max_pooling2d_2597/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42749074Ќ
#conv2d_2598/StatefulPartitionedCallStatefulPartitionedCall%dropout_2597/PartitionedCall:output:0conv2d_2598_42749076conv2d_2598_42749078*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2598_layer_call_and_return_conditional_losses_42748969џ
"max_pooling2d_2598/PartitionedCallPartitionedCall,conv2d_2598/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2598_layer_call_and_return_conditional_losses_42748859ђ
dropout_2598/PartitionedCallPartitionedCall+max_pooling2d_2598/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42749086­
#conv2d_2599/StatefulPartitionedCallStatefulPartitionedCall%dropout_2598/PartitionedCall:output:0conv2d_2599_42749088conv2d_2599_42749090*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2599_layer_call_and_return_conditional_losses_42749001
"max_pooling2d_2599/PartitionedCallPartitionedCall,conv2d_2599/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2599_layer_call_and_return_conditional_losses_42748871ѓ
dropout_2599/PartitionedCallPartitionedCall+max_pooling2d_2599/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749098
,global_average_pooling2d_649/PartitionedCallPartitionedCall%dropout_2599/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_global_average_pooling2d_649_layer_call_and_return_conditional_losses_42748884ѓ
flatten_649/PartitionedCallPartitionedCall5global_average_pooling2d_649/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_flatten_649_layer_call_and_return_conditional_losses_42749029
!dense_649/StatefulPartitionedCallStatefulPartitionedCall$flatten_649/PartitionedCall:output:0dense_649_42749102dense_649_42749104*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_649_layer_call_and_return_conditional_losses_42749042y
IdentityIdentity*dense_649/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp$^conv2d_2596/StatefulPartitionedCall$^conv2d_2597/StatefulPartitionedCall$^conv2d_2598/StatefulPartitionedCall$^conv2d_2599/StatefulPartitionedCall"^dense_649/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ-: : : : : : : : : : 2J
#conv2d_2596/StatefulPartitionedCall#conv2d_2596/StatefulPartitionedCall2J
#conv2d_2597/StatefulPartitionedCall#conv2d_2597/StatefulPartitionedCall2J
#conv2d_2598/StatefulPartitionedCall#conv2d_2598/StatefulPartitionedCall2J
#conv2d_2599/StatefulPartitionedCall#conv2d_2599/StatefulPartitionedCall2F
!dense_649/StatefulPartitionedCall!dense_649/StatefulPartitionedCall:b ^
/
_output_shapes
:џџџџџџџџџ-
+
_user_specified_nameconv2d_2596_input
ў?
а
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749596

inputsD
*conv2d_2596_conv2d_readvariableop_resource:9
+conv2d_2596_biasadd_readvariableop_resource:D
*conv2d_2597_conv2d_readvariableop_resource: 9
+conv2d_2597_biasadd_readvariableop_resource: D
*conv2d_2598_conv2d_readvariableop_resource: @9
+conv2d_2598_biasadd_readvariableop_resource:@E
*conv2d_2599_conv2d_readvariableop_resource:@:
+conv2d_2599_biasadd_readvariableop_resource:	;
(dense_649_matmul_readvariableop_resource:	7
)dense_649_biasadd_readvariableop_resource:
identityЂ"conv2d_2596/BiasAdd/ReadVariableOpЂ!conv2d_2596/Conv2D/ReadVariableOpЂ"conv2d_2597/BiasAdd/ReadVariableOpЂ!conv2d_2597/Conv2D/ReadVariableOpЂ"conv2d_2598/BiasAdd/ReadVariableOpЂ!conv2d_2598/Conv2D/ReadVariableOpЂ"conv2d_2599/BiasAdd/ReadVariableOpЂ!conv2d_2599/Conv2D/ReadVariableOpЂ dense_649/BiasAdd/ReadVariableOpЂdense_649/MatMul/ReadVariableOp
!conv2d_2596/Conv2D/ReadVariableOpReadVariableOp*conv2d_2596_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Б
conv2d_2596/Conv2DConv2Dinputs)conv2d_2596/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ-*
paddingSAME*
strides

"conv2d_2596/BiasAdd/ReadVariableOpReadVariableOp+conv2d_2596_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ё
conv2d_2596/BiasAddBiasAddconv2d_2596/Conv2D:output:0*conv2d_2596/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ-p
conv2d_2596/ReluReluconv2d_2596/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ-Б
max_pooling2d_2596/MaxPoolMaxPoolconv2d_2596/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides

dropout_2596/IdentityIdentity#max_pooling2d_2596/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
!conv2d_2597/Conv2D/ReadVariableOpReadVariableOp*conv2d_2597_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
conv2d_2597/Conv2DConv2Ddropout_2596/Identity:output:0)conv2d_2597/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

"conv2d_2597/BiasAdd/ReadVariableOpReadVariableOp+conv2d_2597_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ё
conv2d_2597/BiasAddBiasAddconv2d_2597/Conv2D:output:0*conv2d_2597/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ p
conv2d_2597/ReluReluconv2d_2597/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Б
max_pooling2d_2597/MaxPoolMaxPoolconv2d_2597/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides

dropout_2597/IdentityIdentity#max_pooling2d_2597/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
!conv2d_2598/Conv2D/ReadVariableOpReadVariableOp*conv2d_2598_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
conv2d_2598/Conv2DConv2Ddropout_2597/Identity:output:0)conv2d_2598/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

"conv2d_2598/BiasAdd/ReadVariableOpReadVariableOp+conv2d_2598_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ё
conv2d_2598/BiasAddBiasAddconv2d_2598/Conv2D:output:0*conv2d_2598/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@p
conv2d_2598/ReluReluconv2d_2598/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Б
max_pooling2d_2598/MaxPoolMaxPoolconv2d_2598/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides

dropout_2598/IdentityIdentity#max_pooling2d_2598/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
!conv2d_2599/Conv2D/ReadVariableOpReadVariableOp*conv2d_2599_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ъ
conv2d_2599/Conv2DConv2Ddropout_2598/Identity:output:0)conv2d_2599/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

"conv2d_2599/BiasAdd/ReadVariableOpReadVariableOp+conv2d_2599_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ђ
conv2d_2599/BiasAddBiasAddconv2d_2599/Conv2D:output:0*conv2d_2599/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџq
conv2d_2599/ReluReluconv2d_2599/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџВ
max_pooling2d_2599/MaxPoolMaxPoolconv2d_2599/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides

dropout_2599/IdentityIdentity#max_pooling2d_2599/MaxPool:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
3global_average_pooling2d_649/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      К
!global_average_pooling2d_649/MeanMeandropout_2599/Identity:output:0<global_average_pooling2d_649/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
flatten_649/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_649/ReshapeReshape*global_average_pooling2d_649/Mean:output:0flatten_649/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_649/MatMul/ReadVariableOpReadVariableOp(dense_649_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_649/MatMulMatMulflatten_649/Reshape:output:0'dense_649/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_649/BiasAdd/ReadVariableOpReadVariableOp)dense_649_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_649/BiasAddBiasAdddense_649/MatMul:product:0(dense_649/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџj
dense_649/SoftmaxSoftmaxdense_649/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
IdentityIdentitydense_649/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЏ
NoOpNoOp#^conv2d_2596/BiasAdd/ReadVariableOp"^conv2d_2596/Conv2D/ReadVariableOp#^conv2d_2597/BiasAdd/ReadVariableOp"^conv2d_2597/Conv2D/ReadVariableOp#^conv2d_2598/BiasAdd/ReadVariableOp"^conv2d_2598/Conv2D/ReadVariableOp#^conv2d_2599/BiasAdd/ReadVariableOp"^conv2d_2599/Conv2D/ReadVariableOp!^dense_649/BiasAdd/ReadVariableOp ^dense_649/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ-: : : : : : : : : : 2H
"conv2d_2596/BiasAdd/ReadVariableOp"conv2d_2596/BiasAdd/ReadVariableOp2F
!conv2d_2596/Conv2D/ReadVariableOp!conv2d_2596/Conv2D/ReadVariableOp2H
"conv2d_2597/BiasAdd/ReadVariableOp"conv2d_2597/BiasAdd/ReadVariableOp2F
!conv2d_2597/Conv2D/ReadVariableOp!conv2d_2597/Conv2D/ReadVariableOp2H
"conv2d_2598/BiasAdd/ReadVariableOp"conv2d_2598/BiasAdd/ReadVariableOp2F
!conv2d_2598/Conv2D/ReadVariableOp!conv2d_2598/Conv2D/ReadVariableOp2H
"conv2d_2599/BiasAdd/ReadVariableOp"conv2d_2599/BiasAdd/ReadVariableOp2F
!conv2d_2599/Conv2D/ReadVariableOp!conv2d_2599/Conv2D/ReadVariableOp2D
 dense_649/BiasAdd/ReadVariableOp dense_649/BiasAdd/ReadVariableOp2B
dense_649/MatMul/ReadVariableOpdense_649/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ-
 
_user_specified_nameinputs
а
K
/__inference_dropout_2599_layer_call_fn_42749807

inputs
identityС
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749098i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
т

i
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42749762

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
в
Y
%__inference__update_step_xla_42703000
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
:
"
_user_specified_name
gradient

l
P__inference_max_pooling2d_2596_layer_call_and_return_conditional_losses_42748835

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ў
M
%__inference__update_step_xla_42703045
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
Л
v
Z__inference_global_average_pooling2d_649_layer_call_and_return_conditional_losses_42749835

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


I__inference_conv2d_2599_layer_call_and_return_conditional_losses_42749787

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
§
h
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42749653

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџc

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н
e
I__inference_flatten_649_layer_call_and_return_conditional_losses_42749029

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

l
P__inference_max_pooling2d_2598_layer_call_and_return_conditional_losses_42749740

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
b
а
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749545

inputsD
*conv2d_2596_conv2d_readvariableop_resource:9
+conv2d_2596_biasadd_readvariableop_resource:D
*conv2d_2597_conv2d_readvariableop_resource: 9
+conv2d_2597_biasadd_readvariableop_resource: D
*conv2d_2598_conv2d_readvariableop_resource: @9
+conv2d_2598_biasadd_readvariableop_resource:@E
*conv2d_2599_conv2d_readvariableop_resource:@:
+conv2d_2599_biasadd_readvariableop_resource:	;
(dense_649_matmul_readvariableop_resource:	7
)dense_649_biasadd_readvariableop_resource:
identityЂ"conv2d_2596/BiasAdd/ReadVariableOpЂ!conv2d_2596/Conv2D/ReadVariableOpЂ"conv2d_2597/BiasAdd/ReadVariableOpЂ!conv2d_2597/Conv2D/ReadVariableOpЂ"conv2d_2598/BiasAdd/ReadVariableOpЂ!conv2d_2598/Conv2D/ReadVariableOpЂ"conv2d_2599/BiasAdd/ReadVariableOpЂ!conv2d_2599/Conv2D/ReadVariableOpЂ dense_649/BiasAdd/ReadVariableOpЂdense_649/MatMul/ReadVariableOp
!conv2d_2596/Conv2D/ReadVariableOpReadVariableOp*conv2d_2596_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Б
conv2d_2596/Conv2DConv2Dinputs)conv2d_2596/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ-*
paddingSAME*
strides

"conv2d_2596/BiasAdd/ReadVariableOpReadVariableOp+conv2d_2596_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ё
conv2d_2596/BiasAddBiasAddconv2d_2596/Conv2D:output:0*conv2d_2596/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ-p
conv2d_2596/ReluReluconv2d_2596/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ-Б
max_pooling2d_2596/MaxPoolMaxPoolconv2d_2596/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
_
dropout_2596/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ѓ
dropout_2596/dropout/MulMul#max_pooling2d_2596/MaxPool:output:0#dropout_2596/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ{
dropout_2596/dropout/ShapeShape#max_pooling2d_2596/MaxPool:output:0*
T0*
_output_shapes
::эЯК
1dropout_2596/dropout/random_uniform/RandomUniformRandomUniform#dropout_2596/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed*h
#dropout_2596/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>е
!dropout_2596/dropout/GreaterEqualGreaterEqual:dropout_2596/dropout/random_uniform/RandomUniform:output:0,dropout_2596/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџa
dropout_2596/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Я
dropout_2596/dropout/SelectV2SelectV2%dropout_2596/dropout/GreaterEqual:z:0dropout_2596/dropout/Mul:z:0%dropout_2596/dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
!conv2d_2597/Conv2D/ReadVariableOpReadVariableOp*conv2d_2597_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0б
conv2d_2597/Conv2DConv2D&dropout_2596/dropout/SelectV2:output:0)conv2d_2597/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

"conv2d_2597/BiasAdd/ReadVariableOpReadVariableOp+conv2d_2597_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ё
conv2d_2597/BiasAddBiasAddconv2d_2597/Conv2D:output:0*conv2d_2597/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ p
conv2d_2597/ReluReluconv2d_2597/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Б
max_pooling2d_2597/MaxPoolMaxPoolconv2d_2597/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
_
dropout_2597/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ѓ
dropout_2597/dropout/MulMul#max_pooling2d_2597/MaxPool:output:0#dropout_2597/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ {
dropout_2597/dropout/ShapeShape#max_pooling2d_2597/MaxPool:output:0*
T0*
_output_shapes
::эЯЧ
1dropout_2597/dropout/random_uniform/RandomUniformRandomUniform#dropout_2597/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
dtype0*
seed2*

seed*h
#dropout_2597/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>е
!dropout_2597/dropout/GreaterEqualGreaterEqual:dropout_2597/dropout/random_uniform/RandomUniform:output:0,dropout_2597/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ a
dropout_2597/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Я
dropout_2597/dropout/SelectV2SelectV2%dropout_2597/dropout/GreaterEqual:z:0dropout_2597/dropout/Mul:z:0%dropout_2597/dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 
!conv2d_2598/Conv2D/ReadVariableOpReadVariableOp*conv2d_2598_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0б
conv2d_2598/Conv2DConv2D&dropout_2597/dropout/SelectV2:output:0)conv2d_2598/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

"conv2d_2598/BiasAdd/ReadVariableOpReadVariableOp+conv2d_2598_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ё
conv2d_2598/BiasAddBiasAddconv2d_2598/Conv2D:output:0*conv2d_2598/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@p
conv2d_2598/ReluReluconv2d_2598/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Б
max_pooling2d_2598/MaxPoolMaxPoolconv2d_2598/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
_
dropout_2598/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ѓ
dropout_2598/dropout/MulMul#max_pooling2d_2598/MaxPool:output:0#dropout_2598/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@{
dropout_2598/dropout/ShapeShape#max_pooling2d_2598/MaxPool:output:0*
T0*
_output_shapes
::эЯЧ
1dropout_2598/dropout/random_uniform/RandomUniformRandomUniform#dropout_2598/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype0*
seed2*

seed*h
#dropout_2598/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>е
!dropout_2598/dropout/GreaterEqualGreaterEqual:dropout_2598/dropout/random_uniform/RandomUniform:output:0,dropout_2598/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
dropout_2598/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Я
dropout_2598/dropout/SelectV2SelectV2%dropout_2598/dropout/GreaterEqual:z:0dropout_2598/dropout/Mul:z:0%dropout_2598/dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
!conv2d_2599/Conv2D/ReadVariableOpReadVariableOp*conv2d_2599_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0в
conv2d_2599/Conv2DConv2D&dropout_2598/dropout/SelectV2:output:0)conv2d_2599/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

"conv2d_2599/BiasAdd/ReadVariableOpReadVariableOp+conv2d_2599_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ђ
conv2d_2599/BiasAddBiasAddconv2d_2599/Conv2D:output:0*conv2d_2599/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџq
conv2d_2599/ReluReluconv2d_2599/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџВ
max_pooling2d_2599/MaxPoolMaxPoolconv2d_2599/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
_
dropout_2599/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Є
dropout_2599/dropout/MulMul#max_pooling2d_2599/MaxPool:output:0#dropout_2599/dropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџ{
dropout_2599/dropout/ShapeShape#max_pooling2d_2599/MaxPool:output:0*
T0*
_output_shapes
::эЯШ
1dropout_2599/dropout/random_uniform/RandomUniformRandomUniform#dropout_2599/dropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
dtype0*
seed2*

seed*h
#dropout_2599/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>ж
!dropout_2599/dropout/GreaterEqualGreaterEqual:dropout_2599/dropout/random_uniform/RandomUniform:output:0,dropout_2599/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџa
dropout_2599/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    а
dropout_2599/dropout/SelectV2SelectV2%dropout_2599/dropout/GreaterEqual:z:0dropout_2599/dropout/Mul:z:0%dropout_2599/dropout/Const_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
3global_average_pooling2d_649/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Т
!global_average_pooling2d_649/MeanMean&dropout_2599/dropout/SelectV2:output:0<global_average_pooling2d_649/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
flatten_649/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_649/ReshapeReshape*global_average_pooling2d_649/Mean:output:0flatten_649/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_649/MatMul/ReadVariableOpReadVariableOp(dense_649_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_649/MatMulMatMulflatten_649/Reshape:output:0'dense_649/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_649/BiasAdd/ReadVariableOpReadVariableOp)dense_649_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_649/BiasAddBiasAdddense_649/MatMul:product:0(dense_649/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџj
dense_649/SoftmaxSoftmaxdense_649/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
IdentityIdentitydense_649/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЏ
NoOpNoOp#^conv2d_2596/BiasAdd/ReadVariableOp"^conv2d_2596/Conv2D/ReadVariableOp#^conv2d_2597/BiasAdd/ReadVariableOp"^conv2d_2597/Conv2D/ReadVariableOp#^conv2d_2598/BiasAdd/ReadVariableOp"^conv2d_2598/Conv2D/ReadVariableOp#^conv2d_2599/BiasAdd/ReadVariableOp"^conv2d_2599/Conv2D/ReadVariableOp!^dense_649/BiasAdd/ReadVariableOp ^dense_649/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ-: : : : : : : : : : 2H
"conv2d_2596/BiasAdd/ReadVariableOp"conv2d_2596/BiasAdd/ReadVariableOp2F
!conv2d_2596/Conv2D/ReadVariableOp!conv2d_2596/Conv2D/ReadVariableOp2H
"conv2d_2597/BiasAdd/ReadVariableOp"conv2d_2597/BiasAdd/ReadVariableOp2F
!conv2d_2597/Conv2D/ReadVariableOp!conv2d_2597/Conv2D/ReadVariableOp2H
"conv2d_2598/BiasAdd/ReadVariableOp"conv2d_2598/BiasAdd/ReadVariableOp2F
!conv2d_2598/Conv2D/ReadVariableOp!conv2d_2598/Conv2D/ReadVariableOp2H
"conv2d_2599/BiasAdd/ReadVariableOp"conv2d_2599/BiasAdd/ReadVariableOp2F
!conv2d_2599/Conv2D/ReadVariableOp!conv2d_2599/Conv2D/ReadVariableOp2D
 dense_649/BiasAdd/ReadVariableOp dense_649/BiasAdd/ReadVariableOp2B
dense_649/MatMul/ReadVariableOpdense_649/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ-
 
_user_specified_nameinputs
т

i
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42749648

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџi
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

l
P__inference_max_pooling2d_2598_layer_call_and_return_conditional_losses_42748859

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

h
/__inference_dropout_2597_layer_call_fn_42749688

inputs
identityЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42748956w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
т

i
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42749705

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


I__inference_conv2d_2598_layer_call_and_return_conditional_losses_42748969

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

h
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749098

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:џџџџџџџџџd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЁB
щ
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749049
conv2d_2596_input.
conv2d_2596_42748906:"
conv2d_2596_42748908:.
conv2d_2597_42748938: "
conv2d_2597_42748940: .
conv2d_2598_42748970: @"
conv2d_2598_42748972:@/
conv2d_2599_42749002:@#
conv2d_2599_42749004:	%
dense_649_42749043:	 
dense_649_42749045:
identityЂ#conv2d_2596/StatefulPartitionedCallЂ#conv2d_2597/StatefulPartitionedCallЂ#conv2d_2598/StatefulPartitionedCallЂ#conv2d_2599/StatefulPartitionedCallЂ!dense_649/StatefulPartitionedCallЂ$dropout_2596/StatefulPartitionedCallЂ$dropout_2597/StatefulPartitionedCallЂ$dropout_2598/StatefulPartitionedCallЂ$dropout_2599/StatefulPartitionedCall
#conv2d_2596/StatefulPartitionedCallStatefulPartitionedCallconv2d_2596_inputconv2d_2596_42748906conv2d_2596_42748908*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2596_layer_call_and_return_conditional_losses_42748905џ
"max_pooling2d_2596/PartitionedCallPartitionedCall,conv2d_2596/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2596_layer_call_and_return_conditional_losses_42748835
$dropout_2596/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_2596/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42748924Д
#conv2d_2597/StatefulPartitionedCallStatefulPartitionedCall-dropout_2596/StatefulPartitionedCall:output:0conv2d_2597_42748938conv2d_2597_42748940*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2597_layer_call_and_return_conditional_losses_42748937џ
"max_pooling2d_2597/PartitionedCallPartitionedCall,conv2d_2597/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2597_layer_call_and_return_conditional_losses_42748847Љ
$dropout_2597/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_2597/PartitionedCall:output:0%^dropout_2596/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42748956Д
#conv2d_2598/StatefulPartitionedCallStatefulPartitionedCall-dropout_2597/StatefulPartitionedCall:output:0conv2d_2598_42748970conv2d_2598_42748972*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2598_layer_call_and_return_conditional_losses_42748969џ
"max_pooling2d_2598/PartitionedCallPartitionedCall,conv2d_2598/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2598_layer_call_and_return_conditional_losses_42748859Љ
$dropout_2598/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_2598/PartitionedCall:output:0%^dropout_2597/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42748988Е
#conv2d_2599/StatefulPartitionedCallStatefulPartitionedCall-dropout_2598/StatefulPartitionedCall:output:0conv2d_2599_42749002conv2d_2599_42749004*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2599_layer_call_and_return_conditional_losses_42749001
"max_pooling2d_2599/PartitionedCallPartitionedCall,conv2d_2599/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_max_pooling2d_2599_layer_call_and_return_conditional_losses_42748871Њ
$dropout_2599/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_2599/PartitionedCall:output:0%^dropout_2598/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749020
,global_average_pooling2d_649/PartitionedCallPartitionedCall-dropout_2599/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *c
f^R\
Z__inference_global_average_pooling2d_649_layer_call_and_return_conditional_losses_42748884ѓ
flatten_649/PartitionedCallPartitionedCall5global_average_pooling2d_649/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_flatten_649_layer_call_and_return_conditional_losses_42749029
!dense_649/StatefulPartitionedCallStatefulPartitionedCall$flatten_649/PartitionedCall:output:0dense_649_42749043dense_649_42749045*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dense_649_layer_call_and_return_conditional_losses_42749042y
IdentityIdentity*dense_649/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp$^conv2d_2596/StatefulPartitionedCall$^conv2d_2597/StatefulPartitionedCall$^conv2d_2598/StatefulPartitionedCall$^conv2d_2599/StatefulPartitionedCall"^dense_649/StatefulPartitionedCall%^dropout_2596/StatefulPartitionedCall%^dropout_2597/StatefulPartitionedCall%^dropout_2598/StatefulPartitionedCall%^dropout_2599/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ-: : : : : : : : : : 2J
#conv2d_2596/StatefulPartitionedCall#conv2d_2596/StatefulPartitionedCall2J
#conv2d_2597/StatefulPartitionedCall#conv2d_2597/StatefulPartitionedCall2J
#conv2d_2598/StatefulPartitionedCall#conv2d_2598/StatefulPartitionedCall2J
#conv2d_2599/StatefulPartitionedCall#conv2d_2599/StatefulPartitionedCall2F
!dense_649/StatefulPartitionedCall!dense_649/StatefulPartitionedCall2L
$dropout_2596/StatefulPartitionedCall$dropout_2596/StatefulPartitionedCall2L
$dropout_2597/StatefulPartitionedCall$dropout_2597/StatefulPartitionedCall2L
$dropout_2598/StatefulPartitionedCall$dropout_2598/StatefulPartitionedCall2L
$dropout_2599/StatefulPartitionedCall$dropout_2599/StatefulPartitionedCall:b ^
/
_output_shapes
:џџџџџџџџџ-
+
_user_specified_nameconv2d_2596_input
ї
Ѓ
.__inference_conv2d_2597_layer_call_fn_42749662

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_2597_layer_call_and_return_conditional_losses_42748937w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"ѓ
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ш
serving_defaultД
W
conv2d_2596_inputB
#serving_default_conv2d_2596_input:0џџџџџџџџџ-=
	dense_6490
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:
А
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
н
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias
 !_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
М
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._random_generator"
_tf_keras_layer
н
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias
 7_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
М
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
D_random_generator"
_tf_keras_layer
н
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias
 M_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
М
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Z_random_generator"
_tf_keras_layer
н
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias
 c_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
М
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
p_random_generator"
_tf_keras_layer
Ѕ
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
Р
}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
h
0
 1
52
63
K4
L5
a6
b7
8
9"
trackable_list_wrapper
h
0
 1
52
63
K4
L5
a6
b7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ї
trace_0
trace_1
trace_2
trace_32
1__inference_sequential_649_layer_call_fn_42749173
1__inference_sequential_649_layer_call_fn_42749237
1__inference_sequential_649_layer_call_fn_42749441
1__inference_sequential_649_layer_call_fn_42749466Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
у
trace_0
trace_1
trace_2
trace_32№
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749049
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749108
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749545
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749596Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
иBе
#__inference__wrapped_model_42748829conv2d_2596_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ѓ

_variables
_iterations
_learning_rate
_index_dict

_momentums
_velocities
_update_step_xla"
experimentalOptimizer
-
serving_default"
signature_map
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ъ
trace_02Ы
.__inference_conv2d_2596_layer_call_fn_42749605
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

 trace_02ц
I__inference_conv2d_2596_layer_call_and_return_conditional_losses_42749616
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z trace_0
,:*2conv2d_2596/kernel
:2conv2d_2596/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
ё
Іtrace_02в
5__inference_max_pooling2d_2596_layer_call_fn_42749621
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zІtrace_0

Їtrace_02э
P__inference_max_pooling2d_2596_layer_call_and_return_conditional_losses_42749626
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЇtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
Щ
­trace_0
Ўtrace_12
/__inference_dropout_2596_layer_call_fn_42749631
/__inference_dropout_2596_layer_call_fn_42749636Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z­trace_0zЎtrace_1
џ
Џtrace_0
Аtrace_12Ф
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42749648
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42749653Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЏtrace_0zАtrace_1
"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
ъ
Жtrace_02Ы
.__inference_conv2d_2597_layer_call_fn_42749662
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЖtrace_0

Зtrace_02ц
I__inference_conv2d_2597_layer_call_and_return_conditional_losses_42749673
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЗtrace_0
,:* 2conv2d_2597/kernel
: 2conv2d_2597/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
ё
Нtrace_02в
5__inference_max_pooling2d_2597_layer_call_fn_42749678
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zНtrace_0

Оtrace_02э
P__inference_max_pooling2d_2597_layer_call_and_return_conditional_losses_42749683
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zОtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
Щ
Фtrace_0
Хtrace_12
/__inference_dropout_2597_layer_call_fn_42749688
/__inference_dropout_2597_layer_call_fn_42749693Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zФtrace_0zХtrace_1
џ
Цtrace_0
Чtrace_12Ф
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42749705
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42749710Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЦtrace_0zЧtrace_1
"
_generic_user_object
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
ъ
Эtrace_02Ы
.__inference_conv2d_2598_layer_call_fn_42749719
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЭtrace_0

Юtrace_02ц
I__inference_conv2d_2598_layer_call_and_return_conditional_losses_42749730
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЮtrace_0
,:* @2conv2d_2598/kernel
:@2conv2d_2598/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
ё
дtrace_02в
5__inference_max_pooling2d_2598_layer_call_fn_42749735
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zдtrace_0

еtrace_02э
P__inference_max_pooling2d_2598_layer_call_and_return_conditional_losses_42749740
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zеtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
Щ
лtrace_0
мtrace_12
/__inference_dropout_2598_layer_call_fn_42749745
/__inference_dropout_2598_layer_call_fn_42749750Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zлtrace_0zмtrace_1
џ
нtrace_0
оtrace_12Ф
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42749762
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42749767Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zнtrace_0zоtrace_1
"
_generic_user_object
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
ъ
фtrace_02Ы
.__inference_conv2d_2599_layer_call_fn_42749776
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zфtrace_0

хtrace_02ц
I__inference_conv2d_2599_layer_call_and_return_conditional_losses_42749787
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zхtrace_0
-:+@2conv2d_2599/kernel
:2conv2d_2599/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
ё
ыtrace_02в
5__inference_max_pooling2d_2599_layer_call_fn_42749792
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zыtrace_0

ьtrace_02э
P__inference_max_pooling2d_2599_layer_call_and_return_conditional_losses_42749797
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zьtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
Щ
ђtrace_0
ѓtrace_12
/__inference_dropout_2599_layer_call_fn_42749802
/__inference_dropout_2599_layer_call_fn_42749807Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zђtrace_0zѓtrace_1
џ
єtrace_0
ѕtrace_12Ф
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749819
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749824Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zєtrace_0zѕtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
іnon_trainable_variables
їlayers
јmetrics
 љlayer_regularization_losses
њlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
ћ
ћtrace_02м
?__inference_global_average_pooling2d_649_layer_call_fn_42749829
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zћtrace_0

ќtrace_02ї
Z__inference_global_average_pooling2d_649_layer_call_and_return_conditional_losses_42749835
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zќtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
§non_trainable_variables
ўlayers
џmetrics
 layer_regularization_losses
layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
ъ
trace_02Ы
.__inference_flatten_649_layer_call_fn_42749840
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ц
I__inference_flatten_649_layer_call_and_return_conditional_losses_42749846
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ш
trace_02Щ
,__inference_dense_649_layer_call_fn_42749855
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ф
G__inference_dense_649_layer_call_and_return_conditional_losses_42749866
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
#:!	2dense_649/kernel
:2dense_649/bias
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
1__inference_sequential_649_layer_call_fn_42749173conv2d_2596_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
1__inference_sequential_649_layer_call_fn_42749237conv2d_2596_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
1__inference_sequential_649_layer_call_fn_42749441inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
1__inference_sequential_649_layer_call_fn_42749466inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749049conv2d_2596_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749108conv2d_2596_input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749545inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749596inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
 20"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
p
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
p
0
1
2
3
4
5
6
7
8
 9"
trackable_list_wrapper
г
Ёtrace_0
Ђtrace_1
Ѓtrace_2
Єtrace_3
Ѕtrace_4
Іtrace_5
Їtrace_6
Јtrace_7
Љtrace_8
Њtrace_92И
%__inference__update_step_xla_42703000
%__inference__update_step_xla_42703005
%__inference__update_step_xla_42703010
%__inference__update_step_xla_42703015
%__inference__update_step_xla_42703020
%__inference__update_step_xla_42703025
%__inference__update_step_xla_42703030
%__inference__update_step_xla_42703035
%__inference__update_step_xla_42703040
%__inference__update_step_xla_42703045Џ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zЁtrace_0zЂtrace_1zЃtrace_2zЄtrace_3zЅtrace_4zІtrace_5zЇtrace_6zЈtrace_7zЉtrace_8zЊtrace_9
зBд
&__inference_signature_wrapper_42749416conv2d_2596_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_conv2d_2596_layer_call_fn_42749605inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_conv2d_2596_layer_call_and_return_conditional_losses_42749616inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
5__inference_max_pooling2d_2596_layer_call_fn_42749621inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
P__inference_max_pooling2d_2596_layer_call_and_return_conditional_losses_42749626inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
/__inference_dropout_2596_layer_call_fn_42749631inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъBч
/__inference_dropout_2596_layer_call_fn_42749636inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42749648inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42749653inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_conv2d_2597_layer_call_fn_42749662inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_conv2d_2597_layer_call_and_return_conditional_losses_42749673inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
5__inference_max_pooling2d_2597_layer_call_fn_42749678inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
P__inference_max_pooling2d_2597_layer_call_and_return_conditional_losses_42749683inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
/__inference_dropout_2597_layer_call_fn_42749688inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъBч
/__inference_dropout_2597_layer_call_fn_42749693inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42749705inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42749710inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_conv2d_2598_layer_call_fn_42749719inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_conv2d_2598_layer_call_and_return_conditional_losses_42749730inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
5__inference_max_pooling2d_2598_layer_call_fn_42749735inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
P__inference_max_pooling2d_2598_layer_call_and_return_conditional_losses_42749740inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
/__inference_dropout_2598_layer_call_fn_42749745inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъBч
/__inference_dropout_2598_layer_call_fn_42749750inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42749762inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42749767inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_conv2d_2599_layer_call_fn_42749776inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_conv2d_2599_layer_call_and_return_conditional_losses_42749787inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
5__inference_max_pooling2d_2599_layer_call_fn_42749792inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
P__inference_max_pooling2d_2599_layer_call_and_return_conditional_losses_42749797inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
/__inference_dropout_2599_layer_call_fn_42749802inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъBч
/__inference_dropout_2599_layer_call_fn_42749807inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749819inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749824inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
?__inference_global_average_pooling2d_649_layer_call_fn_42749829inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
Z__inference_global_average_pooling2d_649_layer_call_and_return_conditional_losses_42749835inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_flatten_649_layer_call_fn_42749840inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_flatten_649_layer_call_and_return_conditional_losses_42749846inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_dense_649_layer_call_fn_42749855inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_dense_649_layer_call_and_return_conditional_losses_42749866inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
Ћ	variables
Ќ	keras_api

­total

Ўcount"
_tf_keras_metric
c
Џ	variables
А	keras_api

Бtotal

Вcount
Г
_fn_kwargs"
_tf_keras_metric
1:/2Adam/m/conv2d_2596/kernel
1:/2Adam/v/conv2d_2596/kernel
#:!2Adam/m/conv2d_2596/bias
#:!2Adam/v/conv2d_2596/bias
1:/ 2Adam/m/conv2d_2597/kernel
1:/ 2Adam/v/conv2d_2597/kernel
#:! 2Adam/m/conv2d_2597/bias
#:! 2Adam/v/conv2d_2597/bias
1:/ @2Adam/m/conv2d_2598/kernel
1:/ @2Adam/v/conv2d_2598/kernel
#:!@2Adam/m/conv2d_2598/bias
#:!@2Adam/v/conv2d_2598/bias
2:0@2Adam/m/conv2d_2599/kernel
2:0@2Adam/v/conv2d_2599/kernel
$:"2Adam/m/conv2d_2599/bias
$:"2Adam/v/conv2d_2599/bias
(:&	2Adam/m/dense_649/kernel
(:&	2Adam/v/dense_649/kernel
!:2Adam/m/dense_649/bias
!:2Adam/v/dense_649/bias
№Bэ
%__inference__update_step_xla_42703000gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
%__inference__update_step_xla_42703005gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
%__inference__update_step_xla_42703010gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
%__inference__update_step_xla_42703015gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
%__inference__update_step_xla_42703020gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
%__inference__update_step_xla_42703025gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
%__inference__update_step_xla_42703030gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
%__inference__update_step_xla_42703035gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
%__inference__update_step_xla_42703040gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
%__inference__update_step_xla_42703045gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
­0
Ў1"
trackable_list_wrapper
.
Ћ	variables"
_generic_user_object
:  (2total
:  (2count
0
Б0
В1"
trackable_list_wrapper
.
Џ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЇ
%__inference__update_step_xla_42703000~xЂu
nЂk
!
gradient
<9	%Ђ"
њ

p
` VariableSpec 
`родКЫ?
Њ "
 
%__inference__update_step_xla_42703005f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`ЙрСЫЫ?
Њ "
 Ї
%__inference__update_step_xla_42703010~xЂu
nЂk
!
gradient 
<9	%Ђ"
њ 

p
` VariableSpec 
` ЋъйсЫ?
Њ "
 
%__inference__update_step_xla_42703015f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
`РъйсЫ?
Њ "
 Ї
%__inference__update_step_xla_42703020~xЂu
nЂk
!
gradient @
<9	%Ђ"
њ @

p
` VariableSpec 
` сЫ?
Њ "
 
%__inference__update_step_xla_42703025f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`рдЏКЫ?
Њ "
 Њ
%__inference__update_step_xla_42703030zЂw
pЂm
"
gradient@
=:	&Ђ#
њ@

p
` VariableSpec 
`рМЯЫ?
Њ "
 
%__inference__update_step_xla_42703035hbЂ_
XЂU

gradient
1.	Ђ
њ

p
` VariableSpec 
` ЯЫ?
Њ "
 
%__inference__update_step_xla_42703040pjЂg
`Ђ]

gradient	
52	Ђ
њ	

p
` VariableSpec 
`рѕдсЫ?
Њ "
 
%__inference__update_step_xla_42703045f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` џЪЫ?
Њ "
 Б
#__inference__wrapped_model_42748829 56KLabBЂ?
8Ђ5
30
conv2d_2596_inputџџџџџџџџџ-
Њ "5Њ2
0
	dense_649# 
	dense_649џџџџџџџџџР
I__inference_conv2d_2596_layer_call_and_return_conditional_losses_42749616s 7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ-
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ-
 
.__inference_conv2d_2596_layer_call_fn_42749605h 7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ-
Њ ")&
unknownџџџџџџџџџ-Р
I__inference_conv2d_2597_layer_call_and_return_conditional_losses_42749673s567Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ 
 
.__inference_conv2d_2597_layer_call_fn_42749662h567Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ")&
unknownџџџџџџџџџ Р
I__inference_conv2d_2598_layer_call_and_return_conditional_losses_42749730sKL7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@
 
.__inference_conv2d_2598_layer_call_fn_42749719hKL7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ ")&
unknownџџџџџџџџџ@С
I__inference_conv2d_2599_layer_call_and_return_conditional_losses_42749787tab7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 
.__inference_conv2d_2599_layer_call_fn_42749776iab7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "*'
unknownџџџџџџџџџБ
G__inference_dense_649_layer_call_and_return_conditional_losses_42749866f0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
,__inference_dense_649_layer_call_fn_42749855[0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџС
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42749648s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 С
J__inference_dropout_2596_layer_call_and_return_conditional_losses_42749653s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 
/__inference_dropout_2596_layer_call_fn_42749631h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ ")&
unknownџџџџџџџџџ
/__inference_dropout_2596_layer_call_fn_42749636h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ ")&
unknownџџџџџџџџџС
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42749705s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ 
 С
J__inference_dropout_2597_layer_call_and_return_conditional_losses_42749710s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ 
 
/__inference_dropout_2597_layer_call_fn_42749688h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ ")&
unknownџџџџџџџџџ 
/__inference_dropout_2597_layer_call_fn_42749693h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ ")&
unknownџџџџџџџџџ С
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42749762s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@
 С
J__inference_dropout_2598_layer_call_and_return_conditional_losses_42749767s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@
 
/__inference_dropout_2598_layer_call_fn_42749745h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ ")&
unknownџџџџџџџџџ@
/__inference_dropout_2598_layer_call_fn_42749750h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ ")&
unknownџџџџџџџџџ@У
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749819u<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 У
J__inference_dropout_2599_layer_call_and_return_conditional_losses_42749824u<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 
/__inference_dropout_2599_layer_call_fn_42749802j<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p
Њ "*'
unknownџџџџџџџџџ
/__inference_dropout_2599_layer_call_fn_42749807j<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p 
Њ "*'
unknownџџџџџџџџџЎ
I__inference_flatten_649_layer_call_and_return_conditional_losses_42749846a0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
.__inference_flatten_649_layer_call_fn_42749840V0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџъ
Z__inference_global_average_pooling2d_649_layer_call_and_return_conditional_losses_42749835RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџџџџџџџџџџ
 Ф
?__inference_global_average_pooling2d_649_layer_call_fn_42749829RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "*'
unknownџџџџџџџџџџџџџџџџџџњ
P__inference_max_pooling2d_2596_layer_call_and_return_conditional_losses_42749626ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 д
5__inference_max_pooling2d_2596_layer_call_fn_42749621RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџњ
P__inference_max_pooling2d_2597_layer_call_and_return_conditional_losses_42749683ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 д
5__inference_max_pooling2d_2597_layer_call_fn_42749678RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџњ
P__inference_max_pooling2d_2598_layer_call_and_return_conditional_losses_42749740ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 д
5__inference_max_pooling2d_2598_layer_call_fn_42749735RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџњ
P__inference_max_pooling2d_2599_layer_call_and_return_conditional_losses_42749797ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 д
5__inference_max_pooling2d_2599_layer_call_fn_42749792RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџй
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749049 56KLabJЂG
@Ђ=
30
conv2d_2596_inputџџџџџџџџџ-
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 й
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749108 56KLabJЂG
@Ђ=
30
conv2d_2596_inputџџџџџџџџџ-
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Э
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749545} 56KLab?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ-
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Э
L__inference_sequential_649_layer_call_and_return_conditional_losses_42749596} 56KLab?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ-
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 В
1__inference_sequential_649_layer_call_fn_42749173} 56KLabJЂG
@Ђ=
30
conv2d_2596_inputџџџџџџџџџ-
p

 
Њ "!
unknownџџџџџџџџџВ
1__inference_sequential_649_layer_call_fn_42749237} 56KLabJЂG
@Ђ=
30
conv2d_2596_inputџџџџџџџџџ-
p 

 
Њ "!
unknownџџџџџџџџџЇ
1__inference_sequential_649_layer_call_fn_42749441r 56KLab?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ-
p

 
Њ "!
unknownџџџџџџџџџЇ
1__inference_sequential_649_layer_call_fn_42749466r 56KLab?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ-
p 

 
Њ "!
unknownџџџџџџџџџЩ
&__inference_signature_wrapper_42749416 56KLabWЂT
Ђ 
MЊJ
H
conv2d_2596_input30
conv2d_2596_inputџџџџџџџџџ-"5Њ2
0
	dense_649# 
	dense_649џџџџџџџџџ