÷Û
»
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
®
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
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
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018èè
¢
%Adam/review_classifier/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/review_classifier/dense_2/bias/v

9Adam/review_classifier/dense_2/bias/v/Read/ReadVariableOpReadVariableOp%Adam/review_classifier/dense_2/bias/v*
_output_shapes
:*
dtype0
ª
'Adam/review_classifier/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'Adam/review_classifier/dense_2/kernel/v
£
;Adam/review_classifier/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/review_classifier/dense_2/kernel/v*
_output_shapes

:*
dtype0
¢
%Adam/review_classifier/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/review_classifier/dense_1/bias/v

9Adam/review_classifier/dense_1/bias/v/Read/ReadVariableOpReadVariableOp%Adam/review_classifier/dense_1/bias/v*
_output_shapes
:*
dtype0
ª
'Adam/review_classifier/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *8
shared_name)'Adam/review_classifier/dense_1/kernel/v
£
;Adam/review_classifier/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/review_classifier/dense_1/kernel/v*
_output_shapes

: *
dtype0
¾
3Adam/review_classifier/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/review_classifier/layer_normalization_1/beta/v
·
GAdam/review_classifier/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp3Adam/review_classifier/layer_normalization_1/beta/v*
_output_shapes
: *
dtype0
À
4Adam/review_classifier/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64Adam/review_classifier/layer_normalization_1/gamma/v
¹
HAdam/review_classifier/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp4Adam/review_classifier/layer_normalization_1/gamma/v*
_output_shapes
: *
dtype0
º
1Adam/review_classifier/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31Adam/review_classifier/layer_normalization/beta/v
³
EAdam/review_classifier/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp1Adam/review_classifier/layer_normalization/beta/v*
_output_shapes
: *
dtype0
¼
2Adam/review_classifier/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42Adam/review_classifier/layer_normalization/gamma/v
µ
FAdam/review_classifier/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp2Adam/review_classifier/layer_normalization/gamma/v*
_output_shapes
: *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
: *
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:  *
dtype0
È
8Adam/review_classifier/attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/review_classifier/attention/attention_output/bias/v
Á
LAdam/review_classifier/attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOp8Adam/review_classifier/attention/attention_output/bias/v*
_output_shapes
: *
dtype0
Ô
:Adam/review_classifier/attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *K
shared_name<:Adam/review_classifier/attention/attention_output/kernel/v
Í
NAdam/review_classifier/attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp:Adam/review_classifier/attention/attention_output/kernel/v*"
_output_shapes
:  *
dtype0
¶
-Adam/review_classifier/attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *>
shared_name/-Adam/review_classifier/attention/value/bias/v
¯
AAdam/review_classifier/attention/value/bias/v/Read/ReadVariableOpReadVariableOp-Adam/review_classifier/attention/value/bias/v*
_output_shapes

: *
dtype0
¾
/Adam/review_classifier/attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *@
shared_name1/Adam/review_classifier/attention/value/kernel/v
·
CAdam/review_classifier/attention/value/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/review_classifier/attention/value/kernel/v*"
_output_shapes
:  *
dtype0
²
+Adam/review_classifier/attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *<
shared_name-+Adam/review_classifier/attention/key/bias/v
«
?Adam/review_classifier/attention/key/bias/v/Read/ReadVariableOpReadVariableOp+Adam/review_classifier/attention/key/bias/v*
_output_shapes

: *
dtype0
º
-Adam/review_classifier/attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/review_classifier/attention/key/kernel/v
³
AAdam/review_classifier/attention/key/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/review_classifier/attention/key/kernel/v*"
_output_shapes
:  *
dtype0
¶
-Adam/review_classifier/attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *>
shared_name/-Adam/review_classifier/attention/query/bias/v
¯
AAdam/review_classifier/attention/query/bias/v/Read/ReadVariableOpReadVariableOp-Adam/review_classifier/attention/query/bias/v*
_output_shapes

: *
dtype0
¾
/Adam/review_classifier/attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *@
shared_name1/Adam/review_classifier/attention/query/kernel/v
·
CAdam/review_classifier/attention/query/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/review_classifier/attention/query/kernel/v*"
_output_shapes
:  *
dtype0
¯
)Adam/review_classifier/p_emb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È *:
shared_name+)Adam/review_classifier/p_emb/embeddings/v
¨
=Adam/review_classifier/p_emb/embeddings/v/Read/ReadVariableOpReadVariableOp)Adam/review_classifier/p_emb/embeddings/v*
_output_shapes
:	È *
dtype0
°
)Adam/review_classifier/t_emb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *:
shared_name+)Adam/review_classifier/t_emb/embeddings/v
©
=Adam/review_classifier/t_emb/embeddings/v/Read/ReadVariableOpReadVariableOp)Adam/review_classifier/t_emb/embeddings/v* 
_output_shapes
:
  *
dtype0
¢
%Adam/review_classifier/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/review_classifier/dense_2/bias/m

9Adam/review_classifier/dense_2/bias/m/Read/ReadVariableOpReadVariableOp%Adam/review_classifier/dense_2/bias/m*
_output_shapes
:*
dtype0
ª
'Adam/review_classifier/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'Adam/review_classifier/dense_2/kernel/m
£
;Adam/review_classifier/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/review_classifier/dense_2/kernel/m*
_output_shapes

:*
dtype0
¢
%Adam/review_classifier/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/review_classifier/dense_1/bias/m

9Adam/review_classifier/dense_1/bias/m/Read/ReadVariableOpReadVariableOp%Adam/review_classifier/dense_1/bias/m*
_output_shapes
:*
dtype0
ª
'Adam/review_classifier/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *8
shared_name)'Adam/review_classifier/dense_1/kernel/m
£
;Adam/review_classifier/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/review_classifier/dense_1/kernel/m*
_output_shapes

: *
dtype0
¾
3Adam/review_classifier/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/review_classifier/layer_normalization_1/beta/m
·
GAdam/review_classifier/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp3Adam/review_classifier/layer_normalization_1/beta/m*
_output_shapes
: *
dtype0
À
4Adam/review_classifier/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64Adam/review_classifier/layer_normalization_1/gamma/m
¹
HAdam/review_classifier/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp4Adam/review_classifier/layer_normalization_1/gamma/m*
_output_shapes
: *
dtype0
º
1Adam/review_classifier/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31Adam/review_classifier/layer_normalization/beta/m
³
EAdam/review_classifier/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp1Adam/review_classifier/layer_normalization/beta/m*
_output_shapes
: *
dtype0
¼
2Adam/review_classifier/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42Adam/review_classifier/layer_normalization/gamma/m
µ
FAdam/review_classifier/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp2Adam/review_classifier/layer_normalization/gamma/m*
_output_shapes
: *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
: *
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:  *
dtype0
È
8Adam/review_classifier/attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/review_classifier/attention/attention_output/bias/m
Á
LAdam/review_classifier/attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOp8Adam/review_classifier/attention/attention_output/bias/m*
_output_shapes
: *
dtype0
Ô
:Adam/review_classifier/attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *K
shared_name<:Adam/review_classifier/attention/attention_output/kernel/m
Í
NAdam/review_classifier/attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp:Adam/review_classifier/attention/attention_output/kernel/m*"
_output_shapes
:  *
dtype0
¶
-Adam/review_classifier/attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *>
shared_name/-Adam/review_classifier/attention/value/bias/m
¯
AAdam/review_classifier/attention/value/bias/m/Read/ReadVariableOpReadVariableOp-Adam/review_classifier/attention/value/bias/m*
_output_shapes

: *
dtype0
¾
/Adam/review_classifier/attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *@
shared_name1/Adam/review_classifier/attention/value/kernel/m
·
CAdam/review_classifier/attention/value/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/review_classifier/attention/value/kernel/m*"
_output_shapes
:  *
dtype0
²
+Adam/review_classifier/attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *<
shared_name-+Adam/review_classifier/attention/key/bias/m
«
?Adam/review_classifier/attention/key/bias/m/Read/ReadVariableOpReadVariableOp+Adam/review_classifier/attention/key/bias/m*
_output_shapes

: *
dtype0
º
-Adam/review_classifier/attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *>
shared_name/-Adam/review_classifier/attention/key/kernel/m
³
AAdam/review_classifier/attention/key/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/review_classifier/attention/key/kernel/m*"
_output_shapes
:  *
dtype0
¶
-Adam/review_classifier/attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *>
shared_name/-Adam/review_classifier/attention/query/bias/m
¯
AAdam/review_classifier/attention/query/bias/m/Read/ReadVariableOpReadVariableOp-Adam/review_classifier/attention/query/bias/m*
_output_shapes

: *
dtype0
¾
/Adam/review_classifier/attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *@
shared_name1/Adam/review_classifier/attention/query/kernel/m
·
CAdam/review_classifier/attention/query/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/review_classifier/attention/query/kernel/m*"
_output_shapes
:  *
dtype0
¯
)Adam/review_classifier/p_emb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È *:
shared_name+)Adam/review_classifier/p_emb/embeddings/m
¨
=Adam/review_classifier/p_emb/embeddings/m/Read/ReadVariableOpReadVariableOp)Adam/review_classifier/p_emb/embeddings/m*
_output_shapes
:	È *
dtype0
°
)Adam/review_classifier/t_emb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *:
shared_name+)Adam/review_classifier/t_emb/embeddings/m
©
=Adam/review_classifier/t_emb/embeddings/m/Read/ReadVariableOpReadVariableOp)Adam/review_classifier/t_emb/embeddings/m* 
_output_shapes
:
  *
dtype0
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
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

review_classifier/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name review_classifier/dense_2/bias

2review_classifier/dense_2/bias/Read/ReadVariableOpReadVariableOpreview_classifier/dense_2/bias*
_output_shapes
:*
dtype0

 review_classifier/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" review_classifier/dense_2/kernel

4review_classifier/dense_2/kernel/Read/ReadVariableOpReadVariableOp review_classifier/dense_2/kernel*
_output_shapes

:*
dtype0

review_classifier/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name review_classifier/dense_1/bias

2review_classifier/dense_1/bias/Read/ReadVariableOpReadVariableOpreview_classifier/dense_1/bias*
_output_shapes
:*
dtype0

 review_classifier/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" review_classifier/dense_1/kernel

4review_classifier/dense_1/kernel/Read/ReadVariableOpReadVariableOp review_classifier/dense_1/kernel*
_output_shapes

: *
dtype0
°
,review_classifier/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,review_classifier/layer_normalization_1/beta
©
@review_classifier/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp,review_classifier/layer_normalization_1/beta*
_output_shapes
: *
dtype0
²
-review_classifier/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-review_classifier/layer_normalization_1/gamma
«
Areview_classifier/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp-review_classifier/layer_normalization_1/gamma*
_output_shapes
: *
dtype0
¬
*review_classifier/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*review_classifier/layer_normalization/beta
¥
>review_classifier/layer_normalization/beta/Read/ReadVariableOpReadVariableOp*review_classifier/layer_normalization/beta*
_output_shapes
: *
dtype0
®
+review_classifier/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+review_classifier/layer_normalization/gamma
§
?review_classifier/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp+review_classifier/layer_normalization/gamma*
_output_shapes
: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:  *
dtype0
º
1review_classifier/attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31review_classifier/attention/attention_output/bias
³
Ereview_classifier/attention/attention_output/bias/Read/ReadVariableOpReadVariableOp1review_classifier/attention/attention_output/bias*
_output_shapes
: *
dtype0
Æ
3review_classifier/attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *D
shared_name53review_classifier/attention/attention_output/kernel
¿
Greview_classifier/attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp3review_classifier/attention/attention_output/kernel*"
_output_shapes
:  *
dtype0
¨
&review_classifier/attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *7
shared_name(&review_classifier/attention/value/bias
¡
:review_classifier/attention/value/bias/Read/ReadVariableOpReadVariableOp&review_classifier/attention/value/bias*
_output_shapes

: *
dtype0
°
(review_classifier/attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *9
shared_name*(review_classifier/attention/value/kernel
©
<review_classifier/attention/value/kernel/Read/ReadVariableOpReadVariableOp(review_classifier/attention/value/kernel*"
_output_shapes
:  *
dtype0
¤
$review_classifier/attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *5
shared_name&$review_classifier/attention/key/bias

8review_classifier/attention/key/bias/Read/ReadVariableOpReadVariableOp$review_classifier/attention/key/bias*
_output_shapes

: *
dtype0
¬
&review_classifier/attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *7
shared_name(&review_classifier/attention/key/kernel
¥
:review_classifier/attention/key/kernel/Read/ReadVariableOpReadVariableOp&review_classifier/attention/key/kernel*"
_output_shapes
:  *
dtype0
¨
&review_classifier/attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *7
shared_name(&review_classifier/attention/query/bias
¡
:review_classifier/attention/query/bias/Read/ReadVariableOpReadVariableOp&review_classifier/attention/query/bias*
_output_shapes

: *
dtype0
°
(review_classifier/attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *9
shared_name*(review_classifier/attention/query/kernel
©
<review_classifier/attention/query/kernel/Read/ReadVariableOpReadVariableOp(review_classifier/attention/query/kernel*"
_output_shapes
:  *
dtype0
¡
"review_classifier/p_emb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È *3
shared_name$"review_classifier/p_emb/embeddings

6review_classifier/p_emb/embeddings/Read/ReadVariableOpReadVariableOp"review_classifier/p_emb/embeddings*
_output_shapes
:	È *
dtype0
¢
"review_classifier/t_emb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *3
shared_name$"review_classifier/t_emb/embeddings

6review_classifier/t_emb/embeddings/Read/ReadVariableOpReadVariableOp"review_classifier/t_emb/embeddings* 
_output_shapes
:
  *
dtype0

NoOpNoOp
­
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*À¬
valueµ¬B±¬ B©¬
ò
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	token_emb
	pos_emb

att
ffn

layernorm1

layernorm2
dropout1
dropout2
pool
dropout3
	dense
dropout4
	stars
	optimizer

signatures*

0
1
2
3
4
5
6
7
8
 9
!10
"11
#12
$13
%14
&15
'16
(17
)18
*19*

0
1
2
3
4
5
6
7
8
 9
!10
"11
#12
$13
%14
&15
'16
(17
)18
*19*
	
+0* 
°
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
1trace_0
2trace_1
3trace_2
4trace_3* 
6
5trace_0
6trace_1
7trace_2
8trace_3* 
* 
 
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

embeddings*
 
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

embeddings*
ù
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
K_query_dense
L
_key_dense
M_value_dense
N_softmax
O_dropout_layer
P_output_dense*
·
Qlayer_with_weights-0
Qlayer-0
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses*
¯
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^axis
	#gamma
$beta*
¯
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
eaxis
	%gamma
&beta*
¥
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
l_random_generator* 
¥
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
s_random_generator* 

t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses* 
¦
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
_random_generator* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

'kernel
(bias*
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

)kernel
*bias*
Ù
	iter
beta_1
beta_2

decay
learning_ratemëmìmímîmïmðmñmòmó mô!mõ"mö#m÷$mø%mù&mú'mû(mü)mý*mþvÿvvvvvvvv v!v"v#v$v%v&v'v(v)v*v*

serving_default* 
b\
VARIABLE_VALUE"review_classifier/t_emb/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"review_classifier/p_emb/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(review_classifier/attention/query/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&review_classifier/attention/query/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&review_classifier/attention/key/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$review_classifier/attention/key/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(review_classifier/attention/value/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&review_classifier/attention/value/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3review_classifier/attention/attention_output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1review_classifier/attention/attention_output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
dense/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+review_classifier/layer_normalization/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*review_classifier/layer_normalization/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-review_classifier/layer_normalization_1/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,review_classifier/layer_normalization_1/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE review_classifier/dense_1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEreview_classifier/dense_1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE review_classifier/dense_2/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEreview_classifier/dense_2/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*

trace_0* 
* 
b
0
	1

2
3
4
5
6
7
8
9
10
11
12*

0
1*
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

0*

0*
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

¢trace_0* 

£trace_0* 

0*

0*
* 

¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

©trace_0* 

ªtrace_0* 
<
0
1
2
3
4
5
6
 7*
<
0
1
2
3
4
5
6
 7*
* 

«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

°trace_0
±trace_1* 

²trace_0
³trace_1* 
ß
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses
ºpartial_output_shape
»full_output_shape

kernel
bias*
ß
¼	variables
½trainable_variables
¾regularization_losses
¿	keras_api
À__call__
+Á&call_and_return_all_conditional_losses
Âpartial_output_shape
Ãfull_output_shape

kernel
bias*
ß
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses
Êpartial_output_shape
Ëfull_output_shape

kernel
bias*

Ì	variables
Ítrainable_variables
Îregularization_losses
Ï	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses* 
¬
Ò	variables
Ótrainable_variables
Ôregularization_losses
Õ	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses
Ø_random_generator* 
ß
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses
ßpartial_output_shape
àfull_output_shape

kernel
 bias*
¬
á	variables
âtrainable_variables
ãregularization_losses
ä	keras_api
å__call__
+æ&call_and_return_all_conditional_losses

!kernel
"bias*

!0
"1*

!0
"1*


ç0* 

ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*
:
ítrace_0
îtrace_1
ïtrace_2
ðtrace_3* 
:
ñtrace_0
òtrace_1
ótrace_2
ôtrace_3* 

#0
$1*

#0
$1*
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

útrace_0* 

ûtrace_0* 
* 

%0
&1*

%0
&1*
* 

ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

¡trace_0
¢trace_1* 

£trace_0
¤trace_1* 
* 

'0
(1*

'0
(1*
	
+0* 

¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ªtrace_0* 

«trace_0* 
* 
* 
* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

±trace_0
²trace_1* 

³trace_0
´trace_1* 
* 

)0
*1*

)0
*1*
* 

µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ºtrace_0* 

»trace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
<
¼	variables
½	keras_api

¾total

¿count*
M
À	variables
Á	keras_api

Âtotal

Ãcount
Ä
_fn_kwargs*
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
.
K0
L1
M2
N3
O4
P5*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses*
* 
* 
* 
* 

0
1*

0
1*
* 

Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
¼	variables
½trainable_variables
¾regularization_losses
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses*
* 
* 
* 
* 

0
1*

0
1*
* 

Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
Ì	variables
Ítrainable_variables
Îregularization_losses
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
Ò	variables
Ótrainable_variables
Ôregularization_losses
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses* 
* 
* 
* 

0
 1*

0
 1*
* 

Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
Ù	variables
Útrainable_variables
Ûregularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses*
* 
* 
* 
* 

!0
"1*

!0
"1*


ç0* 

ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
á	variables
âtrainable_variables
ãregularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses*

ètrace_0* 

étrace_0* 

êtrace_0* 
* 

Q0*
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
	
+0* 
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

¾0
¿1*

¼	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Â0
Ã1*

À	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
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


ç0* 
* 
* 
* 
* 

VARIABLE_VALUE)Adam/review_classifier/t_emb/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/review_classifier/p_emb/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE/Adam/review_classifier/attention/query/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/review_classifier/attention/query/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/review_classifier/attention/key/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/review_classifier/attention/key/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE/Adam/review_classifier/attention/value/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/review_classifier/attention/value/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/review_classifier/attention/attention_output/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/review_classifier/attention/attention_output/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/dense/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/review_classifier/layer_normalization/gamma/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE1Adam/review_classifier/layer_normalization/beta/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE4Adam/review_classifier/layer_normalization_1/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/review_classifier/layer_normalization_1/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/review_classifier/dense_1/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/review_classifier/dense_1/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/review_classifier/dense_2/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/review_classifier/dense_2/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/review_classifier/t_emb/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/review_classifier/p_emb/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE/Adam/review_classifier/attention/query/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/review_classifier/attention/query/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/review_classifier/attention/key/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/review_classifier/attention/key/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE/Adam/review_classifier/attention/value/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/review_classifier/attention/value/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/review_classifier/attention/attention_output/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/review_classifier/attention/attention_output/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/dense/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/review_classifier/layer_normalization/gamma/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE1Adam/review_classifier/layer_normalization/beta/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE4Adam/review_classifier/layer_normalization_1/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/review_classifier/layer_normalization_1/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/review_classifier/dense_1/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/review_classifier/dense_1/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/review_classifier/dense_2/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/review_classifier/dense_2/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
serving_default_input_1Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿÈ
Ô
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1"review_classifier/p_emb/embeddings"review_classifier/t_emb/embeddings(review_classifier/attention/query/kernel&review_classifier/attention/query/bias&review_classifier/attention/key/kernel$review_classifier/attention/key/bias(review_classifier/attention/value/kernel&review_classifier/attention/value/bias3review_classifier/attention/attention_output/kernel1review_classifier/attention/attention_output/bias+review_classifier/layer_normalization/gamma*review_classifier/layer_normalization/betadense/kernel
dense/bias-review_classifier/layer_normalization_1/gamma,review_classifier/layer_normalization_1/beta review_classifier/dense_1/kernelreview_classifier/dense_1/bias review_classifier/dense_2/kernelreview_classifier/dense_2/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_5521241
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
§"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6review_classifier/t_emb/embeddings/Read/ReadVariableOp6review_classifier/p_emb/embeddings/Read/ReadVariableOp<review_classifier/attention/query/kernel/Read/ReadVariableOp:review_classifier/attention/query/bias/Read/ReadVariableOp:review_classifier/attention/key/kernel/Read/ReadVariableOp8review_classifier/attention/key/bias/Read/ReadVariableOp<review_classifier/attention/value/kernel/Read/ReadVariableOp:review_classifier/attention/value/bias/Read/ReadVariableOpGreview_classifier/attention/attention_output/kernel/Read/ReadVariableOpEreview_classifier/attention/attention_output/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp?review_classifier/layer_normalization/gamma/Read/ReadVariableOp>review_classifier/layer_normalization/beta/Read/ReadVariableOpAreview_classifier/layer_normalization_1/gamma/Read/ReadVariableOp@review_classifier/layer_normalization_1/beta/Read/ReadVariableOp4review_classifier/dense_1/kernel/Read/ReadVariableOp2review_classifier/dense_1/bias/Read/ReadVariableOp4review_classifier/dense_2/kernel/Read/ReadVariableOp2review_classifier/dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp=Adam/review_classifier/t_emb/embeddings/m/Read/ReadVariableOp=Adam/review_classifier/p_emb/embeddings/m/Read/ReadVariableOpCAdam/review_classifier/attention/query/kernel/m/Read/ReadVariableOpAAdam/review_classifier/attention/query/bias/m/Read/ReadVariableOpAAdam/review_classifier/attention/key/kernel/m/Read/ReadVariableOp?Adam/review_classifier/attention/key/bias/m/Read/ReadVariableOpCAdam/review_classifier/attention/value/kernel/m/Read/ReadVariableOpAAdam/review_classifier/attention/value/bias/m/Read/ReadVariableOpNAdam/review_classifier/attention/attention_output/kernel/m/Read/ReadVariableOpLAdam/review_classifier/attention/attention_output/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOpFAdam/review_classifier/layer_normalization/gamma/m/Read/ReadVariableOpEAdam/review_classifier/layer_normalization/beta/m/Read/ReadVariableOpHAdam/review_classifier/layer_normalization_1/gamma/m/Read/ReadVariableOpGAdam/review_classifier/layer_normalization_1/beta/m/Read/ReadVariableOp;Adam/review_classifier/dense_1/kernel/m/Read/ReadVariableOp9Adam/review_classifier/dense_1/bias/m/Read/ReadVariableOp;Adam/review_classifier/dense_2/kernel/m/Read/ReadVariableOp9Adam/review_classifier/dense_2/bias/m/Read/ReadVariableOp=Adam/review_classifier/t_emb/embeddings/v/Read/ReadVariableOp=Adam/review_classifier/p_emb/embeddings/v/Read/ReadVariableOpCAdam/review_classifier/attention/query/kernel/v/Read/ReadVariableOpAAdam/review_classifier/attention/query/bias/v/Read/ReadVariableOpAAdam/review_classifier/attention/key/kernel/v/Read/ReadVariableOp?Adam/review_classifier/attention/key/bias/v/Read/ReadVariableOpCAdam/review_classifier/attention/value/kernel/v/Read/ReadVariableOpAAdam/review_classifier/attention/value/bias/v/Read/ReadVariableOpNAdam/review_classifier/attention/attention_output/kernel/v/Read/ReadVariableOpLAdam/review_classifier/attention/attention_output/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpFAdam/review_classifier/layer_normalization/gamma/v/Read/ReadVariableOpEAdam/review_classifier/layer_normalization/beta/v/Read/ReadVariableOpHAdam/review_classifier/layer_normalization_1/gamma/v/Read/ReadVariableOpGAdam/review_classifier/layer_normalization_1/beta/v/Read/ReadVariableOp;Adam/review_classifier/dense_1/kernel/v/Read/ReadVariableOp9Adam/review_classifier/dense_1/bias/v/Read/ReadVariableOp;Adam/review_classifier/dense_2/kernel/v/Read/ReadVariableOp9Adam/review_classifier/dense_2/bias/v/Read/ReadVariableOpConst*R
TinK
I2G	*
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_5522488
¾
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"review_classifier/t_emb/embeddings"review_classifier/p_emb/embeddings(review_classifier/attention/query/kernel&review_classifier/attention/query/bias&review_classifier/attention/key/kernel$review_classifier/attention/key/bias(review_classifier/attention/value/kernel&review_classifier/attention/value/bias3review_classifier/attention/attention_output/kernel1review_classifier/attention/attention_output/biasdense/kernel
dense/bias+review_classifier/layer_normalization/gamma*review_classifier/layer_normalization/beta-review_classifier/layer_normalization_1/gamma,review_classifier/layer_normalization_1/beta review_classifier/dense_1/kernelreview_classifier/dense_1/bias review_classifier/dense_2/kernelreview_classifier/dense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount)Adam/review_classifier/t_emb/embeddings/m)Adam/review_classifier/p_emb/embeddings/m/Adam/review_classifier/attention/query/kernel/m-Adam/review_classifier/attention/query/bias/m-Adam/review_classifier/attention/key/kernel/m+Adam/review_classifier/attention/key/bias/m/Adam/review_classifier/attention/value/kernel/m-Adam/review_classifier/attention/value/bias/m:Adam/review_classifier/attention/attention_output/kernel/m8Adam/review_classifier/attention/attention_output/bias/mAdam/dense/kernel/mAdam/dense/bias/m2Adam/review_classifier/layer_normalization/gamma/m1Adam/review_classifier/layer_normalization/beta/m4Adam/review_classifier/layer_normalization_1/gamma/m3Adam/review_classifier/layer_normalization_1/beta/m'Adam/review_classifier/dense_1/kernel/m%Adam/review_classifier/dense_1/bias/m'Adam/review_classifier/dense_2/kernel/m%Adam/review_classifier/dense_2/bias/m)Adam/review_classifier/t_emb/embeddings/v)Adam/review_classifier/p_emb/embeddings/v/Adam/review_classifier/attention/query/kernel/v-Adam/review_classifier/attention/query/bias/v-Adam/review_classifier/attention/key/kernel/v+Adam/review_classifier/attention/key/bias/v/Adam/review_classifier/attention/value/kernel/v-Adam/review_classifier/attention/value/bias/v:Adam/review_classifier/attention/attention_output/kernel/v8Adam/review_classifier/attention/attention_output/bias/vAdam/dense/kernel/vAdam/dense/bias/v2Adam/review_classifier/layer_normalization/gamma/v1Adam/review_classifier/layer_normalization/beta/v4Adam/review_classifier/layer_normalization_1/gamma/v3Adam/review_classifier/layer_normalization_1/beta/v'Adam/review_classifier/dense_1/kernel/v%Adam/review_classifier/dense_1/bias/v'Adam/review_classifier/dense_2/kernel/v%Adam/review_classifier/dense_2/bias/v*Q
TinJ
H2F*
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_5522705È
ïÐ
¿
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521495
x1
p_emb_embedding_lookup_5521348:	È 2
t_emb_embedding_lookup_5521353:
  K
5attention_query_einsum_einsum_readvariableop_resource:  =
+attention_query_add_readvariableop_resource: I
3attention_key_einsum_einsum_readvariableop_resource:  ;
)attention_key_add_readvariableop_resource: K
5attention_value_einsum_einsum_readvariableop_resource:  =
+attention_value_add_readvariableop_resource: V
@attention_attention_output_einsum_einsum_readvariableop_resource:  D
6attention_attention_output_add_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: D
2sequential_dense_tensordot_readvariableop_resource:  >
0sequential_dense_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity¢-attention/attention_output/add/ReadVariableOp¢7attention/attention_output/einsum/Einsum/ReadVariableOp¢ attention/key/add/ReadVariableOp¢*attention/key/einsum/Einsum/ReadVariableOp¢"attention/query/add/ReadVariableOp¢,attention/query/einsum/Einsum/ReadVariableOp¢"attention/value/add/ReadVariableOp¢,attention/value/einsum/Einsum/ReadVariableOp¢.dense/kernel/Regularizer/Square/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢p_emb/embedding_lookup¢Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢)sequential/dense/Tensordot/ReadVariableOp¢t_emb/embedding_lookupL

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
NotEqualNotEqualxNotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            
strided_sliceStridedSliceNotEqual:z:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

begin_mask	*
end_mask	*
new_axis_mask6
ShapeShapex*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :q
rangeRangerange/start:output:0strided_slice_1:output:0range/delta:output:0*
_output_shapes	
:ÈÉ
p_emb/embedding_lookupResourceGatherp_emb_embedding_lookup_5521348range:output:0*
Tindices0*1
_class'
%#loc:@p_emb/embedding_lookup/5521348*
_output_shapes
:	È *
dtype0©
p_emb/embedding_lookup/IdentityIdentityp_emb/embedding_lookup:output:0*
T0*1
_class'
%#loc:@p_emb/embedding_lookup/5521348*
_output_shapes
:	È 
!p_emb/embedding_lookup/Identity_1Identity(p_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	È É
t_emb/embedding_lookupResourceGathert_emb_embedding_lookup_5521353x*
Tindices0*1
_class'
%#loc:@t_emb/embedding_lookup/5521353*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
dtype0¶
t_emb/embedding_lookup/IdentityIdentityt_emb/embedding_lookup:output:0*
T0*1
_class'
%#loc:@t_emb/embedding_lookup/5521353*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
!t_emb/embedding_lookup/Identity_1Identity(t_emb/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ R
t_emb/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : k
t_emb/NotEqualNotEqualxt_emb/NotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
addAddV2*t_emb/embedding_lookup/Identity_1:output:0*p_emb/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¦
,attention/query/einsum/Einsum/ReadVariableOpReadVariableOp5attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0Ç
attention/query/einsum/EinsumEinsumadd:z:04attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abde
"attention/query/add/ReadVariableOpReadVariableOp+attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype0«
attention/query/addAddV2&attention/query/einsum/Einsum:output:0*attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¢
*attention/key/einsum/Einsum/ReadVariableOpReadVariableOp3attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0Ã
attention/key/einsum/EinsumEinsumadd:z:02attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abde
 attention/key/add/ReadVariableOpReadVariableOp)attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype0¥
attention/key/addAddV2$attention/key/einsum/Einsum:output:0(attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¦
,attention/value/einsum/Einsum/ReadVariableOpReadVariableOp5attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0Ç
attention/value/einsum/EinsumEinsumadd:z:04attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abde
"attention/value/add/ReadVariableOpReadVariableOp+attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype0«
attention/value/addAddV2&attention/value/einsum/Einsum:output:0*attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ T
attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>
attention/MulMulattention/query/add:z:0attention/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¯
attention/einsum/EinsumEinsumattention/key/add:z:0attention/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ*
equationaecd,abcd->acbe
attention/softmax/CastCaststrided_slice:output:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
attention/softmax/subSub attention/softmax/sub/x:output:0attention/softmax/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(knÎ
attention/softmax/mulMulattention/softmax/sub:z:0 attention/softmax/mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
attention/softmax/addAddV2 attention/einsum/Einsum:output:0attention/softmax/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ{
attention/softmax/SoftmaxSoftmaxattention/softmax/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
attention/dropout/IdentityIdentity#attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈÄ
attention/einsum_1/EinsumEinsum#attention/dropout/Identity:output:0attention/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationacbe,aecd->abcd¼
7attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp@attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0ô
(attention/attention_output/einsum/EinsumEinsum"attention/einsum_1/Einsum:output:0?attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabcd,cde->abe 
-attention/attention_output/add/ReadVariableOpReadVariableOp6attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0È
attention/attention_output/addAddV21attention/attention_output/einsum/Einsum:output:05attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ w
dropout/IdentityIdentity"attention/attention_output/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ i
add_1AddV2add:z:0dropout/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¸
 layer_normalization/moments/meanMean	add_1:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ·
-layer_normalization/moments/SquaredDifferenceSquaredDifference	add_1:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:è
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75¾
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
#layer_normalization/batchnorm/mul_1Mul	add_1:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
 sequential/dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : à
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¬
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:½
$sequential/dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ½
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:·
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0°
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ w
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ z
dropout_1/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
add_2AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¼
"layer_normalization_1/moments/meanMean	add_2:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ»
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_2:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:î
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ä
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈª
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
%layer_normalization_1/batchnorm/mul_1Mul	add_2:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¢
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¼
global_average_pooling1d/MeanMean)layer_normalization_1/batchnorm/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
dropout_2/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_1/MatMulMatMuldropout_2/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
dropout_3/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_2/MatMulMatMuldropout_3/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ©
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0²
3review_classifier/dense_1/kernel/Regularizer/SquareSquareJreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 
2review_classifier/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Î
0review_classifier/dense_1/kernel/Regularizer/SumSum7review_classifier/dense_1/kernel/Regularizer/Square:y:0;review_classifier/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: w
2review_classifier/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<Ð
0review_classifier/dense_1/kernel/Regularizer/mulMul;review_classifier/dense_1/kernel/Regularizer/mul/x:output:09review_classifier/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp.^attention/attention_output/add/ReadVariableOp8^attention/attention_output/einsum/Einsum/ReadVariableOp!^attention/key/add/ReadVariableOp+^attention/key/einsum/Einsum/ReadVariableOp#^attention/query/add/ReadVariableOp-^attention/query/einsum/Einsum/ReadVariableOp#^attention/value/add/ReadVariableOp-^attention/value/einsum/Einsum/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp^p_emb/embedding_lookupC^review_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp^t_emb/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : 2^
-attention/attention_output/add/ReadVariableOp-attention/attention_output/add/ReadVariableOp2r
7attention/attention_output/einsum/Einsum/ReadVariableOp7attention/attention_output/einsum/Einsum/ReadVariableOp2D
 attention/key/add/ReadVariableOp attention/key/add/ReadVariableOp2X
*attention/key/einsum/Einsum/ReadVariableOp*attention/key/einsum/Einsum/ReadVariableOp2H
"attention/query/add/ReadVariableOp"attention/query/add/ReadVariableOp2\
,attention/query/einsum/Einsum/ReadVariableOp,attention/query/einsum/Einsum/ReadVariableOp2H
"attention/value/add/ReadVariableOp"attention/value/add/ReadVariableOp2\
,attention/value/einsum/Einsum/ReadVariableOp,attention/value/einsum/Einsum/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp20
p_emb/embedding_lookupp_emb/embedding_lookup2
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpBreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp20
t_emb/embedding_lookupt_emb/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

_user_specified_namex
í
Ô
__inference_loss_fn_0_5521709]
Kreview_classifier_dense_1_kernel_regularizer_square_readvariableop_resource: 
identity¢Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpÎ
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpKreview_classifier_dense_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: *
dtype0²
3review_classifier/dense_1/kernel/Regularizer/SquareSquareJreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 
2review_classifier/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Î
0review_classifier/dense_1/kernel/Regularizer/SumSum7review_classifier/dense_1/kernel/Regularizer/Square:y:0;review_classifier/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: w
2review_classifier/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<Ð
0review_classifier/dense_1/kernel/Regularizer/mulMul;review_classifier/dense_1/kernel/Regularizer/mul/x:output:09review_classifier/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: r
IdentityIdentity4review_classifier/dense_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOpC^review_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpBreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp
%
ª
B__inference_dense_layer_call_and_return_conditional_losses_5520118

inputs3
!tensordot_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp¢.dense/kernel/Regularizer/Square/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ «
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
ë
b
D__inference_dropout_layer_call_and_return_conditional_losses_5522051

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
î
¿
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521686
x1
p_emb_embedding_lookup_5521512:	È 2
t_emb_embedding_lookup_5521517:
  K
5attention_query_einsum_einsum_readvariableop_resource:  =
+attention_query_add_readvariableop_resource: I
3attention_key_einsum_einsum_readvariableop_resource:  ;
)attention_key_add_readvariableop_resource: K
5attention_value_einsum_einsum_readvariableop_resource:  =
+attention_value_add_readvariableop_resource: V
@attention_attention_output_einsum_einsum_readvariableop_resource:  D
6attention_attention_output_add_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: D
2sequential_dense_tensordot_readvariableop_resource:  >
0sequential_dense_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity¢-attention/attention_output/add/ReadVariableOp¢7attention/attention_output/einsum/Einsum/ReadVariableOp¢ attention/key/add/ReadVariableOp¢*attention/key/einsum/Einsum/ReadVariableOp¢"attention/query/add/ReadVariableOp¢,attention/query/einsum/Einsum/ReadVariableOp¢"attention/value/add/ReadVariableOp¢,attention/value/einsum/Einsum/ReadVariableOp¢.dense/kernel/Regularizer/Square/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢p_emb/embedding_lookup¢Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢)sequential/dense/Tensordot/ReadVariableOp¢t_emb/embedding_lookupL

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
NotEqualNotEqualxNotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            
strided_sliceStridedSliceNotEqual:z:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

begin_mask	*
end_mask	*
new_axis_mask6
ShapeShapex*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :q
rangeRangerange/start:output:0strided_slice_1:output:0range/delta:output:0*
_output_shapes	
:ÈÉ
p_emb/embedding_lookupResourceGatherp_emb_embedding_lookup_5521512range:output:0*
Tindices0*1
_class'
%#loc:@p_emb/embedding_lookup/5521512*
_output_shapes
:	È *
dtype0©
p_emb/embedding_lookup/IdentityIdentityp_emb/embedding_lookup:output:0*
T0*1
_class'
%#loc:@p_emb/embedding_lookup/5521512*
_output_shapes
:	È 
!p_emb/embedding_lookup/Identity_1Identity(p_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	È É
t_emb/embedding_lookupResourceGathert_emb_embedding_lookup_5521517x*
Tindices0*1
_class'
%#loc:@t_emb/embedding_lookup/5521517*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
dtype0¶
t_emb/embedding_lookup/IdentityIdentityt_emb/embedding_lookup:output:0*
T0*1
_class'
%#loc:@t_emb/embedding_lookup/5521517*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
!t_emb/embedding_lookup/Identity_1Identity(t_emb/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ R
t_emb/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : k
t_emb/NotEqualNotEqualxt_emb/NotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
addAddV2*t_emb/embedding_lookup/Identity_1:output:0*p_emb/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¦
,attention/query/einsum/Einsum/ReadVariableOpReadVariableOp5attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0Ç
attention/query/einsum/EinsumEinsumadd:z:04attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abde
"attention/query/add/ReadVariableOpReadVariableOp+attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype0«
attention/query/addAddV2&attention/query/einsum/Einsum:output:0*attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¢
*attention/key/einsum/Einsum/ReadVariableOpReadVariableOp3attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0Ã
attention/key/einsum/EinsumEinsumadd:z:02attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abde
 attention/key/add/ReadVariableOpReadVariableOp)attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype0¥
attention/key/addAddV2$attention/key/einsum/Einsum:output:0(attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¦
,attention/value/einsum/Einsum/ReadVariableOpReadVariableOp5attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0Ç
attention/value/einsum/EinsumEinsumadd:z:04attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abde
"attention/value/add/ReadVariableOpReadVariableOp+attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype0«
attention/value/addAddV2&attention/value/einsum/Einsum:output:0*attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ T
attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>
attention/MulMulattention/query/add:z:0attention/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¯
attention/einsum/EinsumEinsumattention/key/add:z:0attention/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ*
equationaecd,abcd->acbe
attention/softmax/CastCaststrided_slice:output:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
attention/softmax/subSub attention/softmax/sub/x:output:0attention/softmax/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(knÎ
attention/softmax/mulMulattention/softmax/sub:z:0 attention/softmax/mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
attention/softmax/addAddV2 attention/einsum/Einsum:output:0attention/softmax/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ{
attention/softmax/SoftmaxSoftmaxattention/softmax/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈÄ
attention/einsum_1/EinsumEinsum#attention/softmax/Softmax:softmax:0attention/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationacbe,aecd->abcd¼
7attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp@attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0ô
(attention/attention_output/einsum/EinsumEinsum"attention/einsum_1/Einsum:output:0?attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabcd,cde->abe 
-attention/attention_output/add/ReadVariableOpReadVariableOp6attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0È
attention/attention_output/addAddV21attention/attention_output/einsum/Einsum:output:05attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout/dropout/MulMul"attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ g
dropout/dropout/ShapeShape"attention/attention_output/add:z:0*
T0*
_output_shapes
:¡
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ã
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ i
add_1AddV2add:z:0dropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¸
 layer_normalization/moments/meanMean	add_1:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ·
-layer_normalization/moments/SquaredDifferenceSquaredDifference	add_1:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:è
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75¾
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
#layer_normalization/batchnorm/mul_1Mul	add_1:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
 sequential/dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : à
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¬
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:½
$sequential/dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0*sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ½
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:·
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0°
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ w
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout_1/dropout/MulMul#sequential/dense/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ j
dropout_1/dropout/ShapeShape#sequential/dense/Relu:activations:0*
T0*
_output_shapes
:¥
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>É
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
add_2AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¼
"layer_normalization_1/moments/meanMean	add_2:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ»
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_2:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:î
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ä
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈª
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
%layer_normalization_1/batchnorm/mul_1Mul	add_2:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¢
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¼
global_average_pooling1d/MeanMean)layer_normalization_1/batchnorm/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout_2/dropout/MulMul&global_average_pooling1d/Mean:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
dropout_2/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ä
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_1/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout_3/dropout/MulMuldense_1/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dropout_3/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
: 
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ä
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_2/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ©
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0²
3review_classifier/dense_1/kernel/Regularizer/SquareSquareJreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 
2review_classifier/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Î
0review_classifier/dense_1/kernel/Regularizer/SumSum7review_classifier/dense_1/kernel/Regularizer/Square:y:0;review_classifier/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: w
2review_classifier/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<Ð
0review_classifier/dense_1/kernel/Regularizer/mulMul;review_classifier/dense_1/kernel/Regularizer/mul/x:output:09review_classifier/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp.^attention/attention_output/add/ReadVariableOp8^attention/attention_output/einsum/Einsum/ReadVariableOp!^attention/key/add/ReadVariableOp+^attention/key/einsum/Einsum/ReadVariableOp#^attention/query/add/ReadVariableOp-^attention/query/einsum/Einsum/ReadVariableOp#^attention/value/add/ReadVariableOp-^attention/value/einsum/Einsum/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp^p_emb/embedding_lookupC^review_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp^t_emb/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : 2^
-attention/attention_output/add/ReadVariableOp-attention/attention_output/add/ReadVariableOp2r
7attention/attention_output/einsum/Einsum/ReadVariableOp7attention/attention_output/einsum/Einsum/ReadVariableOp2D
 attention/key/add/ReadVariableOp attention/key/add/ReadVariableOp2X
*attention/key/einsum/Einsum/ReadVariableOp*attention/key/einsum/Einsum/ReadVariableOp2H
"attention/query/add/ReadVariableOp"attention/query/add/ReadVariableOp2\
,attention/query/einsum/Einsum/ReadVariableOp,attention/query/einsum/Einsum/ReadVariableOp2H
"attention/value/add/ReadVariableOp"attention/value/add/ReadVariableOp2\
,attention/value/einsum/Einsum/ReadVariableOp,attention/value/einsum/Einsum/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp20
p_emb/embedding_lookupp_emb/embedding_lookup2
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpBreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp20
t_emb/embedding_lookupt_emb/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

_user_specified_namex
Å

%__inference_signature_wrapper_5521241
input_1
unknown:	È 
	unknown_0:
  
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:

unknown_17:

unknown_18:
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_5520074o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1
 

õ
D__inference_dense_2_layer_call_and_return_conditional_losses_5520471

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô	
e
F__inference_dropout_3_layer_call_and_return_conditional_losses_5522181

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë0

F__inference_attention_layer_call_and_return_conditional_losses_5521876	
query	
value
attention_mask
A
+query_einsum_einsum_readvariableop_resource:  3
!query_add_readvariableop_resource: ?
)key_einsum_einsum_readvariableop_resource:  1
key_add_readvariableop_resource: A
+value_einsum_einsum_readvariableop_resource:  3
!value_add_readvariableop_resource: L
6attention_output_einsum_einsum_readvariableop_resource:  :
,attention_output_add_readvariableop_resource: 
identity

identity_1¢#attention_output/add/ReadVariableOp¢-attention_output/einsum/Einsum/ReadVariableOp¢key/add/ReadVariableOp¢ key/einsum/Einsum/ReadVariableOp¢query/add/ReadVariableOp¢"query/einsum/Einsum/ReadVariableOp¢value/add/ReadVariableOp¢"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

: *
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

: *
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

: *
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ*
equationaecd,abcd->acben
softmax/CastCastattention_mask*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈR
softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
softmax/subSubsoftmax/sub/x:output:0softmax/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈR
softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(knÎv
softmax/mulMulsoftmax/sub:z:0softmax/mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈy
softmax/addAddV2einsum/Einsum:output:0softmax/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈg
softmax/SoftmaxSoftmaxsoftmax/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ¦
einsum_1/EinsumEinsumsoftmax/Softmax:softmax:0value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0Ö
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0ª
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ t

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈØ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 

_user_specified_namevalue:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(
_user_specified_nameattention_mask
ñ
 
7__inference_layer_normalization_1_layer_call_fn_5522014

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5520416t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
à1

F__inference_attention_layer_call_and_return_conditional_losses_5521834	
query	
value
attention_mask
A
+query_einsum_einsum_readvariableop_resource:  3
!query_add_readvariableop_resource: ?
)key_einsum_einsum_readvariableop_resource:  1
key_add_readvariableop_resource: A
+value_einsum_einsum_readvariableop_resource:  3
!value_add_readvariableop_resource: L
6attention_output_einsum_einsum_readvariableop_resource:  :
,attention_output_add_readvariableop_resource: 
identity

identity_1¢#attention_output/add/ReadVariableOp¢-attention_output/einsum/Einsum/ReadVariableOp¢key/add/ReadVariableOp¢ key/einsum/Einsum/ReadVariableOp¢query/add/ReadVariableOp¢"query/einsum/Einsum/ReadVariableOp¢value/add/ReadVariableOp¢"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

: *
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

: *
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

: *
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ*
equationaecd,abcd->acben
softmax/CastCastattention_mask*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈR
softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
softmax/subSubsoftmax/sub/x:output:0softmax/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈR
softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(knÎv
softmax/mulMulsoftmax/sub:z:0softmax/mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈy
softmax/addAddV2einsum/Einsum:output:0softmax/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈg
softmax/SoftmaxSoftmaxsoftmax/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈs
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ¦
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0Ö
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0ª
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ t

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈØ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 

_user_specified_namevalue:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(
_user_specified_nameattention_mask

ú
G__inference_sequential_layer_call_and_return_conditional_losses_5520205
dense_input
dense_5520193:  
dense_5520195: 
identity¢dense/StatefulPartitionedCall¢.dense/kernel/Regularizer/Square/ReadVariableOpô
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_5520193dense_5520195*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_5520118|
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5520193*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
NoOpNoOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:Y U
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
%
_user_specified_namedense_input
í

5__inference_layer_normalization_layer_call_fn_5521983

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_layer_call_and_return_conditional_losses_5520375t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
ÿ

R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5520416

inputs3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ h
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ w
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ w
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ g
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs

¬
__inference_loss_fn_1_5522258I
7dense_kernel_regularizer_square_readvariableop_resource:  
identity¢.dense/kernel/Regularizer/Square/ReadVariableOp¦
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentity dense/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp
í
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_5522078

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
Å

)__inference_dense_1_layer_call_fn_5522137

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_5520447o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¹e

N__inference_review_classifier_layer_call_and_return_conditional_losses_5520910
x 
p_emb_5520838:	È !
t_emb_5520841:
  '
attention_5520847:  #
attention_5520849: '
attention_5520851:  #
attention_5520853: '
attention_5520855:  #
attention_5520857: '
attention_5520859:  
attention_5520861: )
layer_normalization_5520867: )
layer_normalization_5520869: $
sequential_5520872:   
sequential_5520874: +
layer_normalization_1_5520879: +
layer_normalization_1_5520881: !
dense_1_5520886: 
dense_1_5520888:!
dense_2_5520892:
dense_2_5520894:
identity¢!attention/StatefulPartitionedCall¢.dense/kernel/Regularizer/Square/ReadVariableOp¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢+layer_normalization/StatefulPartitionedCall¢-layer_normalization_1/StatefulPartitionedCall¢p_emb/StatefulPartitionedCall¢Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp¢"sequential/StatefulPartitionedCall¢t_emb/StatefulPartitionedCallL

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
NotEqualNotEqualxNotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            
strided_sliceStridedSliceNotEqual:z:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

begin_mask	*
end_mask	*
new_axis_mask6
ShapeShapex*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :q
rangeRangerange/start:output:0strided_slice_1:output:0range/delta:output:0*
_output_shapes	
:ÈÙ
p_emb/StatefulPartitionedCallStatefulPartitionedCallrange:output:0p_emb_5520838*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	È *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_p_emb_layer_call_and_return_conditional_losses_5520263Ù
t_emb/StatefulPartitionedCallStatefulPartitionedCallxt_emb_5520841*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_t_emb_layer_call_and_return_conditional_losses_5520276R
t_emb/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : k
t_emb/NotEqualNotEqualxt_emb/NotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
addAddV2&t_emb/StatefulPartitionedCall:output:0&p_emb/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¿
!attention/StatefulPartitionedCallStatefulPartitionedCalladd:z:0add:z:0strided_slice:output:0attention_5520847attention_5520849attention_5520851attention_5520853attention_5520855attention_5520857attention_5520859attention_5520861*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈÈ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_attention_layer_call_and_return_conditional_losses_5520737ó
dropout/StatefulPartitionedCallStatefulPartitionedCall*attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_5520662x
add_1AddV2add:z:0(dropout/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ª
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall	add_1:z:0layer_normalization_5520867layer_normalization_5520869*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_layer_call_and_return_conditional_losses_5520375±
"sequential/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0sequential_5520872sequential_5520874*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_5520174
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_5520629§
add_2AddV24layer_normalization/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ²
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall	add_2:z:0layer_normalization_1_5520879layer_normalization_1_5520881*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5520416
(global_average_pooling1d/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5520230
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_5520596
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_1_5520886dense_1_5520888*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_5520447
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_5520563
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_2_5520892dense_2_5520894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_5520471
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5520872*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_5520886*
_output_shapes

: *
dtype0²
3review_classifier/dense_1/kernel/Regularizer/SquareSquareJreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 
2review_classifier/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Î
0review_classifier/dense_1/kernel/Regularizer/SumSum7review_classifier/dense_1/kernel/Regularizer/Square:y:0;review_classifier/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: w
2review_classifier/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<Ð
0review_classifier/dense_1/kernel/Regularizer/mulMul;review_classifier/dense_1/kernel/Regularizer/mul/x:output:09review_classifier/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
NoOpNoOp"^attention/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall^p_emb/StatefulPartitionedCallC^review_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall^t_emb/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : 2F
!attention/StatefulPartitionedCall!attention/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2>
p_emb/StatefulPartitionedCallp_emb/StatefulPartitionedCall2
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpBreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2>
t_emb/StatefulPartitionedCallt_emb/StatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

_user_specified_namex
ñ
|
'__inference_p_emb_layer_call_fn_5521732

inputs
unknown:	È 
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	È *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_p_emb_layer_call_and_return_conditional_losses_5520263g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	È `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
	:È: 22
StatefulPartitionedCallStatefulPartitionedCall:C ?

_output_shapes	
:È
 
_user_specified_nameinputs
¸
G
+__inference_dropout_1_layer_call_fn_5522068

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_5520391e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
ö
§&
 __inference__traced_save_5522488
file_prefixA
=savev2_review_classifier_t_emb_embeddings_read_readvariableopA
=savev2_review_classifier_p_emb_embeddings_read_readvariableopG
Csavev2_review_classifier_attention_query_kernel_read_readvariableopE
Asavev2_review_classifier_attention_query_bias_read_readvariableopE
Asavev2_review_classifier_attention_key_kernel_read_readvariableopC
?savev2_review_classifier_attention_key_bias_read_readvariableopG
Csavev2_review_classifier_attention_value_kernel_read_readvariableopE
Asavev2_review_classifier_attention_value_bias_read_readvariableopR
Nsavev2_review_classifier_attention_attention_output_kernel_read_readvariableopP
Lsavev2_review_classifier_attention_attention_output_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableopJ
Fsavev2_review_classifier_layer_normalization_gamma_read_readvariableopI
Esavev2_review_classifier_layer_normalization_beta_read_readvariableopL
Hsavev2_review_classifier_layer_normalization_1_gamma_read_readvariableopK
Gsavev2_review_classifier_layer_normalization_1_beta_read_readvariableop?
;savev2_review_classifier_dense_1_kernel_read_readvariableop=
9savev2_review_classifier_dense_1_bias_read_readvariableop?
;savev2_review_classifier_dense_2_kernel_read_readvariableop=
9savev2_review_classifier_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopH
Dsavev2_adam_review_classifier_t_emb_embeddings_m_read_readvariableopH
Dsavev2_adam_review_classifier_p_emb_embeddings_m_read_readvariableopN
Jsavev2_adam_review_classifier_attention_query_kernel_m_read_readvariableopL
Hsavev2_adam_review_classifier_attention_query_bias_m_read_readvariableopL
Hsavev2_adam_review_classifier_attention_key_kernel_m_read_readvariableopJ
Fsavev2_adam_review_classifier_attention_key_bias_m_read_readvariableopN
Jsavev2_adam_review_classifier_attention_value_kernel_m_read_readvariableopL
Hsavev2_adam_review_classifier_attention_value_bias_m_read_readvariableopY
Usavev2_adam_review_classifier_attention_attention_output_kernel_m_read_readvariableopW
Ssavev2_adam_review_classifier_attention_attention_output_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableopQ
Msavev2_adam_review_classifier_layer_normalization_gamma_m_read_readvariableopP
Lsavev2_adam_review_classifier_layer_normalization_beta_m_read_readvariableopS
Osavev2_adam_review_classifier_layer_normalization_1_gamma_m_read_readvariableopR
Nsavev2_adam_review_classifier_layer_normalization_1_beta_m_read_readvariableopF
Bsavev2_adam_review_classifier_dense_1_kernel_m_read_readvariableopD
@savev2_adam_review_classifier_dense_1_bias_m_read_readvariableopF
Bsavev2_adam_review_classifier_dense_2_kernel_m_read_readvariableopD
@savev2_adam_review_classifier_dense_2_bias_m_read_readvariableopH
Dsavev2_adam_review_classifier_t_emb_embeddings_v_read_readvariableopH
Dsavev2_adam_review_classifier_p_emb_embeddings_v_read_readvariableopN
Jsavev2_adam_review_classifier_attention_query_kernel_v_read_readvariableopL
Hsavev2_adam_review_classifier_attention_query_bias_v_read_readvariableopL
Hsavev2_adam_review_classifier_attention_key_kernel_v_read_readvariableopJ
Fsavev2_adam_review_classifier_attention_key_bias_v_read_readvariableopN
Jsavev2_adam_review_classifier_attention_value_kernel_v_read_readvariableopL
Hsavev2_adam_review_classifier_attention_value_bias_v_read_readvariableopY
Usavev2_adam_review_classifier_attention_attention_output_kernel_v_read_readvariableopW
Ssavev2_adam_review_classifier_attention_attention_output_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableopQ
Msavev2_adam_review_classifier_layer_normalization_gamma_v_read_readvariableopP
Lsavev2_adam_review_classifier_layer_normalization_beta_v_read_readvariableopS
Osavev2_adam_review_classifier_layer_normalization_1_gamma_v_read_readvariableopR
Nsavev2_adam_review_classifier_layer_normalization_1_beta_v_read_readvariableopF
Bsavev2_adam_review_classifier_dense_1_kernel_v_read_readvariableopD
@savev2_adam_review_classifier_dense_1_bias_v_read_readvariableopF
Bsavev2_adam_review_classifier_dense_2_kernel_v_read_readvariableopD
@savev2_adam_review_classifier_dense_2_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
: ¥ 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*Î
valueÄBÁFB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHü
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*¡
valueBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B %
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_review_classifier_t_emb_embeddings_read_readvariableop=savev2_review_classifier_p_emb_embeddings_read_readvariableopCsavev2_review_classifier_attention_query_kernel_read_readvariableopAsavev2_review_classifier_attention_query_bias_read_readvariableopAsavev2_review_classifier_attention_key_kernel_read_readvariableop?savev2_review_classifier_attention_key_bias_read_readvariableopCsavev2_review_classifier_attention_value_kernel_read_readvariableopAsavev2_review_classifier_attention_value_bias_read_readvariableopNsavev2_review_classifier_attention_attention_output_kernel_read_readvariableopLsavev2_review_classifier_attention_attention_output_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopFsavev2_review_classifier_layer_normalization_gamma_read_readvariableopEsavev2_review_classifier_layer_normalization_beta_read_readvariableopHsavev2_review_classifier_layer_normalization_1_gamma_read_readvariableopGsavev2_review_classifier_layer_normalization_1_beta_read_readvariableop;savev2_review_classifier_dense_1_kernel_read_readvariableop9savev2_review_classifier_dense_1_bias_read_readvariableop;savev2_review_classifier_dense_2_kernel_read_readvariableop9savev2_review_classifier_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopDsavev2_adam_review_classifier_t_emb_embeddings_m_read_readvariableopDsavev2_adam_review_classifier_p_emb_embeddings_m_read_readvariableopJsavev2_adam_review_classifier_attention_query_kernel_m_read_readvariableopHsavev2_adam_review_classifier_attention_query_bias_m_read_readvariableopHsavev2_adam_review_classifier_attention_key_kernel_m_read_readvariableopFsavev2_adam_review_classifier_attention_key_bias_m_read_readvariableopJsavev2_adam_review_classifier_attention_value_kernel_m_read_readvariableopHsavev2_adam_review_classifier_attention_value_bias_m_read_readvariableopUsavev2_adam_review_classifier_attention_attention_output_kernel_m_read_readvariableopSsavev2_adam_review_classifier_attention_attention_output_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableopMsavev2_adam_review_classifier_layer_normalization_gamma_m_read_readvariableopLsavev2_adam_review_classifier_layer_normalization_beta_m_read_readvariableopOsavev2_adam_review_classifier_layer_normalization_1_gamma_m_read_readvariableopNsavev2_adam_review_classifier_layer_normalization_1_beta_m_read_readvariableopBsavev2_adam_review_classifier_dense_1_kernel_m_read_readvariableop@savev2_adam_review_classifier_dense_1_bias_m_read_readvariableopBsavev2_adam_review_classifier_dense_2_kernel_m_read_readvariableop@savev2_adam_review_classifier_dense_2_bias_m_read_readvariableopDsavev2_adam_review_classifier_t_emb_embeddings_v_read_readvariableopDsavev2_adam_review_classifier_p_emb_embeddings_v_read_readvariableopJsavev2_adam_review_classifier_attention_query_kernel_v_read_readvariableopHsavev2_adam_review_classifier_attention_query_bias_v_read_readvariableopHsavev2_adam_review_classifier_attention_key_kernel_v_read_readvariableopFsavev2_adam_review_classifier_attention_key_bias_v_read_readvariableopJsavev2_adam_review_classifier_attention_value_kernel_v_read_readvariableopHsavev2_adam_review_classifier_attention_value_bias_v_read_readvariableopUsavev2_adam_review_classifier_attention_attention_output_kernel_v_read_readvariableopSsavev2_adam_review_classifier_attention_attention_output_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopMsavev2_adam_review_classifier_layer_normalization_gamma_v_read_readvariableopLsavev2_adam_review_classifier_layer_normalization_beta_v_read_readvariableopOsavev2_adam_review_classifier_layer_normalization_1_gamma_v_read_readvariableopNsavev2_adam_review_classifier_layer_normalization_1_beta_v_read_readvariableopBsavev2_adam_review_classifier_dense_1_kernel_v_read_readvariableop@savev2_adam_review_classifier_dense_1_bias_v_read_readvariableopBsavev2_adam_review_classifier_dense_2_kernel_v_read_readvariableop@savev2_adam_review_classifier_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *T
dtypesJ
H2F	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ü
_input_shapesÊ
Ç: :
  :	È :  : :  : :  : :  : :  : : : : : : :::: : : : : : : : : :
  :	È :  : :  : :  : :  : :  : : : : : : ::::
  :	È :  : :  : :  : :  : :  : : : : : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
  :%!

_output_shapes
:	È :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :(	$
"
_output_shapes
:  : 


_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
  :%!

_output_shapes
:	È :( $
"
_output_shapes
:  :$! 

_output_shapes

: :("$
"
_output_shapes
:  :$# 

_output_shapes

: :($$
"
_output_shapes
:  :$% 

_output_shapes

: :(&$
"
_output_shapes
:  : '

_output_shapes
: :$( 

_output_shapes

:  : )

_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: : ,

_output_shapes
: : -

_output_shapes
: :$. 

_output_shapes

: : /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::&2"
 
_output_shapes
:
  :%3!

_output_shapes
:	È :(4$
"
_output_shapes
:  :$5 

_output_shapes

: :(6$
"
_output_shapes
:  :$7 

_output_shapes

: :(8$
"
_output_shapes
:  :$9 

_output_shapes

: :(:$
"
_output_shapes
:  : ;

_output_shapes
: :$< 

_output_shapes

:  : =

_output_shapes
: : >

_output_shapes
: : ?

_output_shapes
: : @

_output_shapes
: : A

_output_shapes
: :$B 

_output_shapes

: : C

_output_shapes
::$D 

_output_shapes

:: E

_output_shapes
::F

_output_shapes
: 


e
F__inference_dropout_1_layer_call_and_return_conditional_losses_5522090

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
ë0

F__inference_attention_layer_call_and_return_conditional_losses_5520737	
query	
value
attention_mask
A
+query_einsum_einsum_readvariableop_resource:  3
!query_add_readvariableop_resource: ?
)key_einsum_einsum_readvariableop_resource:  1
key_add_readvariableop_resource: A
+value_einsum_einsum_readvariableop_resource:  3
!value_add_readvariableop_resource: L
6attention_output_einsum_einsum_readvariableop_resource:  :
,attention_output_add_readvariableop_resource: 
identity

identity_1¢#attention_output/add/ReadVariableOp¢-attention_output/einsum/Einsum/ReadVariableOp¢key/add/ReadVariableOp¢ key/einsum/Einsum/ReadVariableOp¢query/add/ReadVariableOp¢"query/einsum/Einsum/ReadVariableOp¢value/add/ReadVariableOp¢"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

: *
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

: *
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

: *
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ*
equationaecd,abcd->acben
softmax/CastCastattention_mask*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈR
softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
softmax/subSubsoftmax/sub/x:output:0softmax/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈR
softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(knÎv
softmax/mulMulsoftmax/sub:z:0softmax/mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈy
softmax/addAddV2einsum/Einsum:output:0softmax/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈg
softmax/SoftmaxSoftmaxsoftmax/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ¦
einsum_1/EinsumEinsumsoftmax/Softmax:softmax:0value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0Ö
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0ª
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ t

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈØ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 

_user_specified_namevalue:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(
_user_specified_nameattention_mask
·_
þ	
N__inference_review_classifier_layer_call_and_return_conditional_losses_5520490
x 
p_emb_5520264:	È !
t_emb_5520277:
  '
attention_5520327:  #
attention_5520329: '
attention_5520331:  #
attention_5520333: '
attention_5520335:  #
attention_5520337: '
attention_5520339:  
attention_5520341: )
layer_normalization_5520376: )
layer_normalization_5520378: $
sequential_5520381:   
sequential_5520383: +
layer_normalization_1_5520417: +
layer_normalization_1_5520419: !
dense_1_5520448: 
dense_1_5520450:!
dense_2_5520472:
dense_2_5520474:
identity¢!attention/StatefulPartitionedCall¢.dense/kernel/Regularizer/Square/ReadVariableOp¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢+layer_normalization/StatefulPartitionedCall¢-layer_normalization_1/StatefulPartitionedCall¢p_emb/StatefulPartitionedCall¢Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp¢"sequential/StatefulPartitionedCall¢t_emb/StatefulPartitionedCallL

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
NotEqualNotEqualxNotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            
strided_sliceStridedSliceNotEqual:z:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

begin_mask	*
end_mask	*
new_axis_mask6
ShapeShapex*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :q
rangeRangerange/start:output:0strided_slice_1:output:0range/delta:output:0*
_output_shapes	
:ÈÙ
p_emb/StatefulPartitionedCallStatefulPartitionedCallrange:output:0p_emb_5520264*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	È *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_p_emb_layer_call_and_return_conditional_losses_5520263Ù
t_emb/StatefulPartitionedCallStatefulPartitionedCallxt_emb_5520277*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_t_emb_layer_call_and_return_conditional_losses_5520276R
t_emb/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : k
t_emb/NotEqualNotEqualxt_emb/NotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
addAddV2&t_emb/StatefulPartitionedCall:output:0&p_emb/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¿
!attention/StatefulPartitionedCallStatefulPartitionedCalladd:z:0add:z:0strided_slice:output:0attention_5520327attention_5520329attention_5520331attention_5520333attention_5520335attention_5520337attention_5520339attention_5520341*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈÈ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_attention_layer_call_and_return_conditional_losses_5520326ã
dropout/PartitionedCallPartitionedCall*attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_5520350p
add_1AddV2add:z:0 dropout/PartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ª
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall	add_1:z:0layer_normalization_5520376layer_normalization_5520378*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_layer_call_and_return_conditional_losses_5520375±
"sequential/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0sequential_5520381sequential_5520383*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_5520131è
dropout_1/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_5520391
add_2AddV24layer_normalization/StatefulPartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ²
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall	add_2:z:0layer_normalization_1_5520417layer_normalization_1_5520419*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5520416
(global_average_pooling1d/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5520230é
dropout_2/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_5520428
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_1_5520448dense_1_5520450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_5520447à
dropout_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_5520458
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_2_5520472dense_2_5520474*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_5520471
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5520381*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_5520448*
_output_shapes

: *
dtype0²
3review_classifier/dense_1/kernel/Regularizer/SquareSquareJreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 
2review_classifier/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Î
0review_classifier/dense_1/kernel/Regularizer/SumSum7review_classifier/dense_1/kernel/Regularizer/Square:y:0;review_classifier/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: w
2review_classifier/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<Ð
0review_classifier/dense_1/kernel/Regularizer/mulMul;review_classifier/dense_1/kernel/Regularizer/mul/x:output:09review_classifier/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
NoOpNoOp"^attention/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall^p_emb/StatefulPartitionedCallC^review_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall^t_emb/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : 2F
!attention/StatefulPartitionedCall!attention/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2>
p_emb/StatefulPartitionedCallp_emb/StatefulPartitionedCall2
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpBreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2>
t_emb/StatefulPartitionedCallt_emb/StatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

_user_specified_namex
Û_


N__inference_review_classifier_layer_call_and_return_conditional_losses_5521087
input_1 
p_emb_5521015:	È !
t_emb_5521018:
  '
attention_5521024:  #
attention_5521026: '
attention_5521028:  #
attention_5521030: '
attention_5521032:  #
attention_5521034: '
attention_5521036:  
attention_5521038: )
layer_normalization_5521044: )
layer_normalization_5521046: $
sequential_5521049:   
sequential_5521051: +
layer_normalization_1_5521056: +
layer_normalization_1_5521058: !
dense_1_5521063: 
dense_1_5521065:!
dense_2_5521069:
dense_2_5521071:
identity¢!attention/StatefulPartitionedCall¢.dense/kernel/Regularizer/Square/ReadVariableOp¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢+layer_normalization/StatefulPartitionedCall¢-layer_normalization_1/StatefulPartitionedCall¢p_emb/StatefulPartitionedCall¢Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp¢"sequential/StatefulPartitionedCall¢t_emb/StatefulPartitionedCallL

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : e
NotEqualNotEqualinput_1NotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            
strided_sliceStridedSliceNotEqual:z:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

begin_mask	*
end_mask	*
new_axis_mask<
ShapeShapeinput_1*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :q
rangeRangerange/start:output:0strided_slice_1:output:0range/delta:output:0*
_output_shapes	
:ÈÙ
p_emb/StatefulPartitionedCallStatefulPartitionedCallrange:output:0p_emb_5521015*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	È *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_p_emb_layer_call_and_return_conditional_losses_5520263ß
t_emb/StatefulPartitionedCallStatefulPartitionedCallinput_1t_emb_5521018*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_t_emb_layer_call_and_return_conditional_losses_5520276R
t_emb/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : q
t_emb/NotEqualNotEqualinput_1t_emb/NotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
addAddV2&t_emb/StatefulPartitionedCall:output:0&p_emb/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¿
!attention/StatefulPartitionedCallStatefulPartitionedCalladd:z:0add:z:0strided_slice:output:0attention_5521024attention_5521026attention_5521028attention_5521030attention_5521032attention_5521034attention_5521036attention_5521038*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈÈ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_attention_layer_call_and_return_conditional_losses_5520326ã
dropout/PartitionedCallPartitionedCall*attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_5520350p
add_1AddV2add:z:0 dropout/PartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ª
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall	add_1:z:0layer_normalization_5521044layer_normalization_5521046*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_layer_call_and_return_conditional_losses_5520375±
"sequential/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0sequential_5521049sequential_5521051*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_5520131è
dropout_1/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_5520391
add_2AddV24layer_normalization/StatefulPartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ²
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall	add_2:z:0layer_normalization_1_5521056layer_normalization_1_5521058*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5520416
(global_average_pooling1d/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5520230é
dropout_2/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_5520428
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_1_5521063dense_1_5521065*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_5520447à
dropout_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_5520458
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_2_5521069dense_2_5521071*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_5520471
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5521049*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_5521063*
_output_shapes

: *
dtype0²
3review_classifier/dense_1/kernel/Regularizer/SquareSquareJreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 
2review_classifier/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Î
0review_classifier/dense_1/kernel/Regularizer/SumSum7review_classifier/dense_1/kernel/Regularizer/Square:y:0;review_classifier/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: w
2review_classifier/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<Ð
0review_classifier/dense_1/kernel/Regularizer/mulMul;review_classifier/dense_1/kernel/Regularizer/mul/x:output:09review_classifier/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
NoOpNoOp"^attention/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall^p_emb/StatefulPartitionedCallC^review_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall^t_emb/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : 2F
!attention/StatefulPartitionedCall!attention/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2>
p_emb/StatefulPartitionedCallp_emb/StatefulPartitionedCall2
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpBreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2>
t_emb/StatefulPartitionedCallt_emb/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1

d
+__inference_dropout_1_layer_call_fn_5522073

inputs
identity¢StatefulPartitionedCallÉ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_5520629t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
ë
b
D__inference_dropout_layer_call_and_return_conditional_losses_5520350

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
´
E
)__inference_dropout_layer_call_fn_5522041

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_5520350e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
Ñ
¢
B__inference_t_emb_layer_call_and_return_conditional_losses_5520276

inputs,
embedding_lookup_5520270:
  
identity¢embedding_lookup¼
embedding_lookupResourceGatherembedding_lookup_5520270inputs*
Tindices0*+
_class!
loc:@embedding_lookup/5520270*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
dtype0¤
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/5520270*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¬(
Ç
G__inference_sequential_layer_call_and_return_conditional_losses_5521974

inputs9
'dense_tensordot_readvariableop_resource:  3
%dense_biasadd_readvariableop_resource: 
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢.dense/kernel/Regularizer/Square/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: _
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ a

dense/ReluReludense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: l
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ·
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
ô	
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_5522128

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
­
º
D__inference_dense_1_layer_call_and_return_conditional_losses_5522154

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0²
3review_classifier/dense_1/kernel/Regularizer/SquareSquareJreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 
2review_classifier/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Î
0review_classifier/dense_1/kernel/Regularizer/SumSum7review_classifier/dense_1/kernel/Regularizer/Square:y:0;review_classifier/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: w
2review_classifier/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<Ð
0review_classifier/dense_1/kernel/Regularizer/mulMul;review_classifier/dense_1/kernel/Regularizer/mul/x:output:09review_classifier/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOpC^review_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpBreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
%
ª
B__inference_dense_layer_call_and_return_conditional_losses_5522247

inputs3
!tensordot_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp¢.dense/kernel/Regularizer/Square/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ «
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
ÿ

R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5522036

inputs3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ h
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ w
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ w
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ g
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs

¡
B__inference_p_emb_layer_call_and_return_conditional_losses_5521741

inputs+
embedding_lookup_5521735:	È 
identity¢embedding_lookup¯
embedding_lookupResourceGatherembedding_lookup_5521735inputs*
Tindices0*+
_class!
loc:@embedding_lookup/5521735*
_output_shapes
:	È *
dtype0
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/5521735*
_output_shapes
:	È u
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	È k
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*
_output_shapes
:	È Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
	:È: 2$
embedding_lookupembedding_lookup:C ?

_output_shapes	
:È
 
_user_specified_nameinputs
Ù
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_5520428

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Õ

'__inference_dense_layer_call_fn_5522210

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_5520118t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs

q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5522101

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
¢
B__inference_t_emb_layer_call_and_return_conditional_losses_5521725

inputs,
embedding_lookup_5521719:
  
identity¢embedding_lookup¼
embedding_lookupResourceGatherembedding_lookup_5521719inputs*
Tindices0*+
_class!
loc:@embedding_lookup/5521719*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
dtype0¤
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/5521719*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ã

+__inference_attention_layer_call_fn_5521791	
query	
value
attention_mask

unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
identity

identity_1¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallqueryvalueattention_maskunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈÈ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_attention_layer_call_and_return_conditional_losses_5520737t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ {

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 

_user_specified_namevalue:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(
_user_specified_nameattention_mask
ý

P__inference_layer_normalization_layer_call_and_return_conditional_losses_5520375

inputs3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ h
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ w
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ w
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ g
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
ö
ó
"__inference__wrapped_model_5520074
input_1C
0review_classifier_p_emb_embedding_lookup_5519939:	È D
0review_classifier_t_emb_embedding_lookup_5519944:
  ]
Greview_classifier_attention_query_einsum_einsum_readvariableop_resource:  O
=review_classifier_attention_query_add_readvariableop_resource: [
Ereview_classifier_attention_key_einsum_einsum_readvariableop_resource:  M
;review_classifier_attention_key_add_readvariableop_resource: ]
Greview_classifier_attention_value_einsum_einsum_readvariableop_resource:  O
=review_classifier_attention_value_add_readvariableop_resource: h
Rreview_classifier_attention_attention_output_einsum_einsum_readvariableop_resource:  V
Hreview_classifier_attention_attention_output_add_readvariableop_resource: Y
Kreview_classifier_layer_normalization_batchnorm_mul_readvariableop_resource: U
Greview_classifier_layer_normalization_batchnorm_readvariableop_resource: V
Dreview_classifier_sequential_dense_tensordot_readvariableop_resource:  P
Breview_classifier_sequential_dense_biasadd_readvariableop_resource: [
Mreview_classifier_layer_normalization_1_batchnorm_mul_readvariableop_resource: W
Ireview_classifier_layer_normalization_1_batchnorm_readvariableop_resource: J
8review_classifier_dense_1_matmul_readvariableop_resource: G
9review_classifier_dense_1_biasadd_readvariableop_resource:J
8review_classifier_dense_2_matmul_readvariableop_resource:G
9review_classifier_dense_2_biasadd_readvariableop_resource:
identity¢?review_classifier/attention/attention_output/add/ReadVariableOp¢Ireview_classifier/attention/attention_output/einsum/Einsum/ReadVariableOp¢2review_classifier/attention/key/add/ReadVariableOp¢<review_classifier/attention/key/einsum/Einsum/ReadVariableOp¢4review_classifier/attention/query/add/ReadVariableOp¢>review_classifier/attention/query/einsum/Einsum/ReadVariableOp¢4review_classifier/attention/value/add/ReadVariableOp¢>review_classifier/attention/value/einsum/Einsum/ReadVariableOp¢0review_classifier/dense_1/BiasAdd/ReadVariableOp¢/review_classifier/dense_1/MatMul/ReadVariableOp¢0review_classifier/dense_2/BiasAdd/ReadVariableOp¢/review_classifier/dense_2/MatMul/ReadVariableOp¢>review_classifier/layer_normalization/batchnorm/ReadVariableOp¢Breview_classifier/layer_normalization/batchnorm/mul/ReadVariableOp¢@review_classifier/layer_normalization_1/batchnorm/ReadVariableOp¢Dreview_classifier/layer_normalization_1/batchnorm/mul/ReadVariableOp¢(review_classifier/p_emb/embedding_lookup¢9review_classifier/sequential/dense/BiasAdd/ReadVariableOp¢;review_classifier/sequential/dense/Tensordot/ReadVariableOp¢(review_classifier/t_emb/embedding_lookup^
review_classifier/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
review_classifier/NotEqualNotEqualinput_1%review_classifier/NotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ~
%review_classifier/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                
'review_classifier/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                
'review_classifier/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            â
review_classifier/strided_sliceStridedSlicereview_classifier/NotEqual:z:0.review_classifier/strided_slice/stack:output:00review_classifier/strided_slice/stack_1:output:00review_classifier/strided_slice/stack_2:output:0*
Index0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

begin_mask	*
end_mask	*
new_axis_maskN
review_classifier/ShapeShapeinput_1*
T0*
_output_shapes
:z
'review_classifier/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿs
)review_classifier/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)review_classifier/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
!review_classifier/strided_slice_1StridedSlice review_classifier/Shape:output:00review_classifier/strided_slice_1/stack:output:02review_classifier/strided_slice_1/stack_1:output:02review_classifier/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
review_classifier/range/startConst*
_output_shapes
: *
dtype0*
value	B : _
review_classifier/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :¹
review_classifier/rangeRange&review_classifier/range/start:output:0*review_classifier/strided_slice_1:output:0&review_classifier/range/delta:output:0*
_output_shapes	
:È
(review_classifier/p_emb/embedding_lookupResourceGather0review_classifier_p_emb_embedding_lookup_5519939 review_classifier/range:output:0*
Tindices0*C
_class9
75loc:@review_classifier/p_emb/embedding_lookup/5519939*
_output_shapes
:	È *
dtype0ß
1review_classifier/p_emb/embedding_lookup/IdentityIdentity1review_classifier/p_emb/embedding_lookup:output:0*
T0*C
_class9
75loc:@review_classifier/p_emb/embedding_lookup/5519939*
_output_shapes
:	È ¥
3review_classifier/p_emb/embedding_lookup/Identity_1Identity:review_classifier/p_emb/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	È 
(review_classifier/t_emb/embedding_lookupResourceGather0review_classifier_t_emb_embedding_lookup_5519944input_1*
Tindices0*C
_class9
75loc:@review_classifier/t_emb/embedding_lookup/5519944*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
dtype0ì
1review_classifier/t_emb/embedding_lookup/IdentityIdentity1review_classifier/t_emb/embedding_lookup:output:0*
T0*C
_class9
75loc:@review_classifier/t_emb/embedding_lookup/5519944*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ²
3review_classifier/t_emb/embedding_lookup/Identity_1Identity:review_classifier/t_emb/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ d
"review_classifier/t_emb/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
 review_classifier/t_emb/NotEqualNotEqualinput_1+review_classifier/t_emb/NotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÑ
review_classifier/addAddV2<review_classifier/t_emb/embedding_lookup/Identity_1:output:0<review_classifier/p_emb/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ Ê
>review_classifier/attention/query/einsum/Einsum/ReadVariableOpReadVariableOpGreview_classifier_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0ý
/review_classifier/attention/query/einsum/EinsumEinsumreview_classifier/add:z:0Freview_classifier/attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abde²
4review_classifier/attention/query/add/ReadVariableOpReadVariableOp=review_classifier_attention_query_add_readvariableop_resource*
_output_shapes

: *
dtype0á
%review_classifier/attention/query/addAddV28review_classifier/attention/query/einsum/Einsum:output:0<review_classifier/attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ Æ
<review_classifier/attention/key/einsum/Einsum/ReadVariableOpReadVariableOpEreview_classifier_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0ù
-review_classifier/attention/key/einsum/EinsumEinsumreview_classifier/add:z:0Dreview_classifier/attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abde®
2review_classifier/attention/key/add/ReadVariableOpReadVariableOp;review_classifier_attention_key_add_readvariableop_resource*
_output_shapes

: *
dtype0Û
#review_classifier/attention/key/addAddV26review_classifier/attention/key/einsum/Einsum:output:0:review_classifier/attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ Ê
>review_classifier/attention/value/einsum/Einsum/ReadVariableOpReadVariableOpGreview_classifier_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0ý
/review_classifier/attention/value/einsum/EinsumEinsumreview_classifier/add:z:0Freview_classifier/attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abde²
4review_classifier/attention/value/add/ReadVariableOpReadVariableOp=review_classifier_attention_value_add_readvariableop_resource*
_output_shapes

: *
dtype0á
%review_classifier/attention/value/addAddV28review_classifier/attention/value/einsum/Einsum:output:0<review_classifier/attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ f
!review_classifier/attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>¸
review_classifier/attention/MulMul)review_classifier/attention/query/add:z:0*review_classifier/attention/Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ å
)review_classifier/attention/einsum/EinsumEinsum'review_classifier/attention/key/add:z:0#review_classifier/attention/Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ*
equationaecd,abcd->acbe¤
(review_classifier/attention/softmax/CastCast(review_classifier/strided_slice:output:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈn
)review_classifier/attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ë
'review_classifier/attention/softmax/subSub2review_classifier/attention/softmax/sub/x:output:0,review_classifier/attention/softmax/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈn
)review_classifier/attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(knÎÊ
'review_classifier/attention/softmax/mulMul+review_classifier/attention/softmax/sub:z:02review_classifier/attention/softmax/mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÍ
'review_classifier/attention/softmax/addAddV22review_classifier/attention/einsum/Einsum:output:0+review_classifier/attention/softmax/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
+review_classifier/attention/softmax/SoftmaxSoftmax+review_classifier/attention/softmax/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ«
,review_classifier/attention/dropout/IdentityIdentity5review_classifier/attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈú
+review_classifier/attention/einsum_1/EinsumEinsum5review_classifier/attention/dropout/Identity:output:0)review_classifier/attention/value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationacbe,aecd->abcdà
Ireview_classifier/attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpRreview_classifier_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0ª
:review_classifier/attention/attention_output/einsum/EinsumEinsum4review_classifier/attention/einsum_1/Einsum:output:0Qreview_classifier/attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabcd,cde->abeÄ
?review_classifier/attention/attention_output/add/ReadVariableOpReadVariableOpHreview_classifier_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0þ
0review_classifier/attention/attention_output/addAddV2Creview_classifier/attention/attention_output/einsum/Einsum:output:0Greview_classifier/attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
"review_classifier/dropout/IdentityIdentity4review_classifier/attention/attention_output/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
review_classifier/add_1AddV2review_classifier/add:z:0+review_classifier/dropout/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
Dreview_classifier/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:î
2review_classifier/layer_normalization/moments/meanMeanreview_classifier/add_1:z:0Mreview_classifier/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(¾
:review_classifier/layer_normalization/moments/StopGradientStopGradient;review_classifier/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈí
?review_classifier/layer_normalization/moments/SquaredDifferenceSquaredDifferencereview_classifier/add_1:z:0Creview_classifier/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
Hreview_classifier/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
6review_classifier/layer_normalization/moments/varianceMeanCreview_classifier/layer_normalization/moments/SquaredDifference:z:0Qreview_classifier/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(z
5review_classifier/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75ô
3review_classifier/layer_normalization/batchnorm/addAddV2?review_classifier/layer_normalization/moments/variance:output:0>review_classifier/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ®
5review_classifier/layer_normalization/batchnorm/RsqrtRsqrt7review_classifier/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÊ
Breview_classifier/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKreview_classifier_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0ø
3review_classifier/layer_normalization/batchnorm/mulMul9review_classifier/layer_normalization/batchnorm/Rsqrt:y:0Jreview_classifier/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ É
5review_classifier/layer_normalization/batchnorm/mul_1Mulreview_classifier/add_1:z:07review_classifier/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ é
5review_classifier/layer_normalization/batchnorm/mul_2Mul;review_classifier/layer_normalization/moments/mean:output:07review_classifier/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ Â
>review_classifier/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGreview_classifier_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0ô
3review_classifier/layer_normalization/batchnorm/subSubFreview_classifier/layer_normalization/batchnorm/ReadVariableOp:value:09review_classifier/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ é
5review_classifier/layer_normalization/batchnorm/add_1AddV29review_classifier/layer_normalization/batchnorm/mul_1:z:07review_classifier/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ À
;review_classifier/sequential/dense/Tensordot/ReadVariableOpReadVariableOpDreview_classifier_sequential_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0{
1review_classifier/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
1review_classifier/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
2review_classifier/sequential/dense/Tensordot/ShapeShape9review_classifier/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:|
:review_classifier/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
5review_classifier/sequential/dense/Tensordot/GatherV2GatherV2;review_classifier/sequential/dense/Tensordot/Shape:output:0:review_classifier/sequential/dense/Tensordot/free:output:0Creview_classifier/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
<review_classifier/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
7review_classifier/sequential/dense/Tensordot/GatherV2_1GatherV2;review_classifier/sequential/dense/Tensordot/Shape:output:0:review_classifier/sequential/dense/Tensordot/axes:output:0Ereview_classifier/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:|
2review_classifier/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ×
1review_classifier/sequential/dense/Tensordot/ProdProd>review_classifier/sequential/dense/Tensordot/GatherV2:output:0;review_classifier/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: ~
4review_classifier/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ý
3review_classifier/sequential/dense/Tensordot/Prod_1Prod@review_classifier/sequential/dense/Tensordot/GatherV2_1:output:0=review_classifier/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: z
8review_classifier/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¨
3review_classifier/sequential/dense/Tensordot/concatConcatV2:review_classifier/sequential/dense/Tensordot/free:output:0:review_classifier/sequential/dense/Tensordot/axes:output:0Areview_classifier/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:â
2review_classifier/sequential/dense/Tensordot/stackPack:review_classifier/sequential/dense/Tensordot/Prod:output:0<review_classifier/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ó
6review_classifier/sequential/dense/Tensordot/transpose	Transpose9review_classifier/layer_normalization/batchnorm/add_1:z:0<review_classifier/sequential/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ó
4review_classifier/sequential/dense/Tensordot/ReshapeReshape:review_classifier/sequential/dense/Tensordot/transpose:y:0;review_classifier/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
3review_classifier/sequential/dense/Tensordot/MatMulMatMul=review_classifier/sequential/dense/Tensordot/Reshape:output:0Creview_classifier/sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
4review_classifier/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: |
:review_classifier/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
5review_classifier/sequential/dense/Tensordot/concat_1ConcatV2>review_classifier/sequential/dense/Tensordot/GatherV2:output:0=review_classifier/sequential/dense/Tensordot/Const_2:output:0Creview_classifier/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:í
,review_classifier/sequential/dense/TensordotReshape=review_classifier/sequential/dense/Tensordot/MatMul:product:0>review_classifier/sequential/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¸
9review_classifier/sequential/dense/BiasAdd/ReadVariableOpReadVariableOpBreview_classifier_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0æ
*review_classifier/sequential/dense/BiasAddBiasAdd5review_classifier/sequential/dense/Tensordot:output:0Areview_classifier/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
'review_classifier/sequential/dense/ReluRelu3review_classifier/sequential/dense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
$review_classifier/dropout_1/IdentityIdentity5review_classifier/sequential/dense/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ Á
review_classifier/add_2AddV29review_classifier/layer_normalization/batchnorm/add_1:z:0-review_classifier/dropout_1/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
Freview_classifier/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ò
4review_classifier/layer_normalization_1/moments/meanMeanreview_classifier/add_2:z:0Oreview_classifier/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(Â
<review_classifier/layer_normalization_1/moments/StopGradientStopGradient=review_classifier/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈñ
Areview_classifier/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencereview_classifier/add_2:z:0Ereview_classifier/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
Jreview_classifier/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¤
8review_classifier/layer_normalization_1/moments/varianceMeanEreview_classifier/layer_normalization_1/moments/SquaredDifference:z:0Sreview_classifier/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(|
7review_classifier/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75ú
5review_classifier/layer_normalization_1/batchnorm/addAddV2Areview_classifier/layer_normalization_1/moments/variance:output:0@review_classifier/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ²
7review_classifier/layer_normalization_1/batchnorm/RsqrtRsqrt9review_classifier/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÎ
Dreview_classifier/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMreview_classifier_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0þ
5review_classifier/layer_normalization_1/batchnorm/mulMul;review_classifier/layer_normalization_1/batchnorm/Rsqrt:y:0Lreview_classifier/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ Í
7review_classifier/layer_normalization_1/batchnorm/mul_1Mulreview_classifier/add_2:z:09review_classifier/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ï
7review_classifier/layer_normalization_1/batchnorm/mul_2Mul=review_classifier/layer_normalization_1/moments/mean:output:09review_classifier/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ Æ
@review_classifier/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpIreview_classifier_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0ú
5review_classifier/layer_normalization_1/batchnorm/subSubHreview_classifier/layer_normalization_1/batchnorm/ReadVariableOp:value:0;review_classifier/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ï
7review_classifier/layer_normalization_1/batchnorm/add_1AddV2;review_classifier/layer_normalization_1/batchnorm/mul_1:z:09review_classifier/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
Areview_classifier/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :ò
/review_classifier/global_average_pooling1d/MeanMean;review_classifier/layer_normalization_1/batchnorm/add_1:z:0Jreview_classifier/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$review_classifier/dropout_2/IdentityIdentity8review_classifier/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
/review_classifier/dense_1/MatMul/ReadVariableOpReadVariableOp8review_classifier_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ä
 review_classifier/dense_1/MatMulMatMul-review_classifier/dropout_2/Identity:output:07review_classifier/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0review_classifier/dense_1/BiasAdd/ReadVariableOpReadVariableOp9review_classifier_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!review_classifier/dense_1/BiasAddBiasAdd*review_classifier/dense_1/MatMul:product:08review_classifier/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
review_classifier/dense_1/ReluRelu*review_classifier/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$review_classifier/dropout_3/IdentityIdentity,review_classifier/dense_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/review_classifier/dense_2/MatMul/ReadVariableOpReadVariableOp8review_classifier_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ä
 review_classifier/dense_2/MatMulMatMul-review_classifier/dropout_3/Identity:output:07review_classifier/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0review_classifier/dense_2/BiasAdd/ReadVariableOpReadVariableOp9review_classifier_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!review_classifier/dense_2/BiasAddBiasAdd*review_classifier/dense_2/MatMul:product:08review_classifier/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!review_classifier/dense_2/SoftmaxSoftmax*review_classifier/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
IdentityIdentity+review_classifier/dense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ	
NoOpNoOp@^review_classifier/attention/attention_output/add/ReadVariableOpJ^review_classifier/attention/attention_output/einsum/Einsum/ReadVariableOp3^review_classifier/attention/key/add/ReadVariableOp=^review_classifier/attention/key/einsum/Einsum/ReadVariableOp5^review_classifier/attention/query/add/ReadVariableOp?^review_classifier/attention/query/einsum/Einsum/ReadVariableOp5^review_classifier/attention/value/add/ReadVariableOp?^review_classifier/attention/value/einsum/Einsum/ReadVariableOp1^review_classifier/dense_1/BiasAdd/ReadVariableOp0^review_classifier/dense_1/MatMul/ReadVariableOp1^review_classifier/dense_2/BiasAdd/ReadVariableOp0^review_classifier/dense_2/MatMul/ReadVariableOp?^review_classifier/layer_normalization/batchnorm/ReadVariableOpC^review_classifier/layer_normalization/batchnorm/mul/ReadVariableOpA^review_classifier/layer_normalization_1/batchnorm/ReadVariableOpE^review_classifier/layer_normalization_1/batchnorm/mul/ReadVariableOp)^review_classifier/p_emb/embedding_lookup:^review_classifier/sequential/dense/BiasAdd/ReadVariableOp<^review_classifier/sequential/dense/Tensordot/ReadVariableOp)^review_classifier/t_emb/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : 2
?review_classifier/attention/attention_output/add/ReadVariableOp?review_classifier/attention/attention_output/add/ReadVariableOp2
Ireview_classifier/attention/attention_output/einsum/Einsum/ReadVariableOpIreview_classifier/attention/attention_output/einsum/Einsum/ReadVariableOp2h
2review_classifier/attention/key/add/ReadVariableOp2review_classifier/attention/key/add/ReadVariableOp2|
<review_classifier/attention/key/einsum/Einsum/ReadVariableOp<review_classifier/attention/key/einsum/Einsum/ReadVariableOp2l
4review_classifier/attention/query/add/ReadVariableOp4review_classifier/attention/query/add/ReadVariableOp2
>review_classifier/attention/query/einsum/Einsum/ReadVariableOp>review_classifier/attention/query/einsum/Einsum/ReadVariableOp2l
4review_classifier/attention/value/add/ReadVariableOp4review_classifier/attention/value/add/ReadVariableOp2
>review_classifier/attention/value/einsum/Einsum/ReadVariableOp>review_classifier/attention/value/einsum/Einsum/ReadVariableOp2d
0review_classifier/dense_1/BiasAdd/ReadVariableOp0review_classifier/dense_1/BiasAdd/ReadVariableOp2b
/review_classifier/dense_1/MatMul/ReadVariableOp/review_classifier/dense_1/MatMul/ReadVariableOp2d
0review_classifier/dense_2/BiasAdd/ReadVariableOp0review_classifier/dense_2/BiasAdd/ReadVariableOp2b
/review_classifier/dense_2/MatMul/ReadVariableOp/review_classifier/dense_2/MatMul/ReadVariableOp2
>review_classifier/layer_normalization/batchnorm/ReadVariableOp>review_classifier/layer_normalization/batchnorm/ReadVariableOp2
Breview_classifier/layer_normalization/batchnorm/mul/ReadVariableOpBreview_classifier/layer_normalization/batchnorm/mul/ReadVariableOp2
@review_classifier/layer_normalization_1/batchnorm/ReadVariableOp@review_classifier/layer_normalization_1/batchnorm/ReadVariableOp2
Dreview_classifier/layer_normalization_1/batchnorm/mul/ReadVariableOpDreview_classifier/layer_normalization_1/batchnorm/mul/ReadVariableOp2T
(review_classifier/p_emb/embedding_lookup(review_classifier/p_emb/embedding_lookup2v
9review_classifier/sequential/dense/BiasAdd/ReadVariableOp9review_classifier/sequential/dense/BiasAdd/ReadVariableOp2z
;review_classifier/sequential/dense/Tensordot/ReadVariableOp;review_classifier/sequential/dense/Tensordot/ReadVariableOp2T
(review_classifier/t_emb/embedding_lookup(review_classifier/t_emb/embedding_lookup:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1

V
:__inference_global_average_pooling1d_layer_call_fn_5522095

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5520230i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß

,__inference_sequential_layer_call_fn_5521900

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_5520174t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
Ù
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_5522169

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
G
+__inference_dropout_2_layer_call_fn_5522106

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_5520428`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
à1

F__inference_attention_layer_call_and_return_conditional_losses_5520326	
query	
value
attention_mask
A
+query_einsum_einsum_readvariableop_resource:  3
!query_add_readvariableop_resource: ?
)key_einsum_einsum_readvariableop_resource:  1
key_add_readvariableop_resource: A
+value_einsum_einsum_readvariableop_resource:  3
!value_add_readvariableop_resource: L
6attention_output_einsum_einsum_readvariableop_resource:  :
,attention_output_add_readvariableop_resource: 
identity

identity_1¢#attention_output/add/ReadVariableOp¢-attention_output/einsum/Einsum/ReadVariableOp¢key/add/ReadVariableOp¢ key/einsum/Einsum/ReadVariableOp¢query/add/ReadVariableOp¢"query/einsum/Einsum/ReadVariableOp¢value/add/ReadVariableOp¢"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0±
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

: *
dtype0
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0­
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

: *
dtype0
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0±
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

: *
dtype0
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>d
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ*
equationaecd,abcd->acben
softmax/CastCastattention_mask*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈR
softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
softmax/subSubsoftmax/sub/x:output:0softmax/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈR
softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(knÎv
softmax/mulMulsoftmax/sub:z:0softmax/mul/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈy
softmax/addAddV2einsum/Einsum:output:0softmax/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈg
softmax/SoftmaxSoftmaxsoftmax/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈs
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ¦
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationacbe,aecd->abcd¨
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype0Ö
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0ª
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ t

Identity_1Identitysoftmax/Softmax:softmax:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈØ
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 

_user_specified_namevalue:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(
_user_specified_nameattention_mask
ÿ
õ
G__inference_sequential_layer_call_and_return_conditional_losses_5520131

inputs
dense_5520119:  
dense_5520121: 
identity¢dense/StatefulPartitionedCall¢.dense/kernel/Regularizer/Square/ReadVariableOpï
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5520119dense_5520121*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_5520118|
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5520119*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
NoOpNoOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
ÿ
õ
G__inference_sequential_layer_call_and_return_conditional_losses_5520174

inputs
dense_5520162:  
dense_5520164: 
identity¢dense/StatefulPartitionedCall¢.dense/kernel/Regularizer/Square/ReadVariableOpï
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5520162dense_5520164*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_5520118|
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5520162*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
NoOpNoOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
í
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_5520391

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs

¡
B__inference_p_emb_layer_call_and_return_conditional_losses_5520263

inputs+
embedding_lookup_5520257:	È 
identity¢embedding_lookup¯
embedding_lookupResourceGatherembedding_lookup_5520257inputs*
Tindices0*+
_class!
loc:@embedding_lookup/5520257*
_output_shapes
:	È *
dtype0
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/5520257*
_output_shapes
:	È u
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	È k
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*
_output_shapes
:	È Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
	:È: 2$
embedding_lookupembedding_lookup:C ?

_output_shapes	
:È
 
_user_specified_nameinputs
¦
}
'__inference_t_emb_layer_call_fn_5521716

inputs
unknown:
  
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_t_emb_layer_call_and_return_conditional_losses_5520276t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ö
d
+__inference_dropout_3_layer_call_fn_5522164

inputs
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_5520563o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ú
G__inference_sequential_layer_call_and_return_conditional_losses_5520220
dense_input
dense_5520208:  
dense_5520210: 
identity¢dense/StatefulPartitionedCall¢.dense/kernel/Regularizer/Square/ReadVariableOpô
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_5520208dense_5520210*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_5520118|
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5520208*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
NoOpNoOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:Y U
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
%
_user_specified_namedense_input
ô	
e
F__inference_dropout_3_layer_call_and_return_conditional_losses_5520563

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_5522116

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ù
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_5520458

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
º
D__inference_dense_1_layer_call_and_return_conditional_losses_5520447

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0²
3review_classifier/dense_1/kernel/Regularizer/SquareSquareJreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 
2review_classifier/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Î
0review_classifier/dense_1/kernel/Regularizer/SumSum7review_classifier/dense_1/kernel/Regularizer/Square:y:0;review_classifier/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: w
2review_classifier/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<Ð
0review_classifier/dense_1/kernel/Regularizer/mulMul;review_classifier/dense_1/kernel/Regularizer/mul/x:output:09review_classifier/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOpC^review_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpBreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

q
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5520230

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
G
+__inference_dropout_3_layer_call_fn_5522159

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_5520458`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß

,__inference_sequential_layer_call_fn_5521891

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_5520131t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
 

õ
D__inference_dense_2_layer_call_and_return_conditional_losses_5522201

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
«
3__inference_review_classifier_layer_call_fn_5520533
input_1
unknown:	È 
	unknown_0:
  
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:

unknown_17:

unknown_18:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_review_classifier_layer_call_and_return_conditional_losses_5520490o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1
í
¥
3__inference_review_classifier_layer_call_fn_5521286
x
unknown:	È 
	unknown_0:
  
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:

unknown_17:

unknown_18:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_review_classifier_layer_call_and_return_conditional_losses_5520490o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

_user_specified_namex
ÿ
«
3__inference_review_classifier_layer_call_fn_5520998
input_1
unknown:	È 
	unknown_0:
  
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:

unknown_17:

unknown_18:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_review_classifier_layer_call_and_return_conditional_losses_5520910o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1
í
¥
3__inference_review_classifier_layer_call_fn_5521331
x
unknown:	È 
	unknown_0:
  
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:

unknown_17:

unknown_18:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_review_classifier_layer_call_and_return_conditional_losses_5520910o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

_user_specified_namex
Ýe

N__inference_review_classifier_layer_call_and_return_conditional_losses_5521176
input_1 
p_emb_5521104:	È !
t_emb_5521107:
  '
attention_5521113:  #
attention_5521115: '
attention_5521117:  #
attention_5521119: '
attention_5521121:  #
attention_5521123: '
attention_5521125:  
attention_5521127: )
layer_normalization_5521133: )
layer_normalization_5521135: $
sequential_5521138:   
sequential_5521140: +
layer_normalization_1_5521145: +
layer_normalization_1_5521147: !
dense_1_5521152: 
dense_1_5521154:!
dense_2_5521158:
dense_2_5521160:
identity¢!attention/StatefulPartitionedCall¢.dense/kernel/Regularizer/Square/ReadVariableOp¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢+layer_normalization/StatefulPartitionedCall¢-layer_normalization_1/StatefulPartitionedCall¢p_emb/StatefulPartitionedCall¢Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp¢"sequential/StatefulPartitionedCall¢t_emb/StatefulPartitionedCallL

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : e
NotEqualNotEqualinput_1NotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            
strided_sliceStridedSliceNotEqual:z:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

begin_mask	*
end_mask	*
new_axis_mask<
ShapeShapeinput_1*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :q
rangeRangerange/start:output:0strided_slice_1:output:0range/delta:output:0*
_output_shapes	
:ÈÙ
p_emb/StatefulPartitionedCallStatefulPartitionedCallrange:output:0p_emb_5521104*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	È *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_p_emb_layer_call_and_return_conditional_losses_5520263ß
t_emb/StatefulPartitionedCallStatefulPartitionedCallinput_1t_emb_5521107*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_t_emb_layer_call_and_return_conditional_losses_5520276R
t_emb/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : q
t_emb/NotEqualNotEqualinput_1t_emb/NotEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
addAddV2&t_emb/StatefulPartitionedCall:output:0&p_emb/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ¿
!attention/StatefulPartitionedCallStatefulPartitionedCalladd:z:0add:z:0strided_slice:output:0attention_5521113attention_5521115attention_5521117attention_5521119attention_5521121attention_5521123attention_5521125attention_5521127*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈÈ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_attention_layer_call_and_return_conditional_losses_5520737ó
dropout/StatefulPartitionedCallStatefulPartitionedCall*attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_5520662x
add_1AddV2add:z:0(dropout/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ª
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall	add_1:z:0layer_normalization_5521133layer_normalization_5521135*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_layer_call_and_return_conditional_losses_5520375±
"sequential/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0sequential_5521138sequential_5521140*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_5520174
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_5520629§
add_2AddV24layer_normalization/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ²
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall	add_2:z:0layer_normalization_1_5521145layer_normalization_1_5521147*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5520416
(global_average_pooling1d/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5520230
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_5520596
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_1_5521152dense_1_5521154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_5520447
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_5520563
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_2_5521158dense_2_5521160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_5520471
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5521138*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_5521152*
_output_shapes

: *
dtype0²
3review_classifier/dense_1/kernel/Regularizer/SquareSquareJreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 
2review_classifier/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Î
0review_classifier/dense_1/kernel/Regularizer/SumSum7review_classifier/dense_1/kernel/Regularizer/Square:y:0;review_classifier/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: w
2review_classifier/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<Ð
0review_classifier/dense_1/kernel/Regularizer/mulMul;review_classifier/dense_1/kernel/Regularizer/mul/x:output:09review_classifier/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
NoOpNoOp"^attention/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall^p_emb/StatefulPartitionedCallC^review_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall^t_emb/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : : : : : : : : : : : : : 2F
!attention/StatefulPartitionedCall!attention/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2>
p_emb/StatefulPartitionedCallp_emb/StatefulPartitionedCall2
Breview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOpBreview_classifier/dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2>
t_emb/StatefulPartitionedCallt_emb/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
!
_user_specified_name	input_1
î

,__inference_sequential_layer_call_fn_5520138
dense_input
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_5520131t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
%
_user_specified_namedense_input


c
D__inference_dropout_layer_call_and_return_conditional_losses_5520662

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs


e
F__inference_dropout_1_layer_call_and_return_conditional_losses_5520629

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
ý

P__inference_layer_normalization_layer_call_and_return_conditional_losses_5522005

inputs3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(r
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:¬
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈb
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ h
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ w
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ w
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ g
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
ö
d
+__inference_dropout_2_layer_call_fn_5522111

inputs
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_5520596o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Å

)__inference_dense_2_layer_call_fn_5522190

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_5520471o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

+__inference_attention_layer_call_fn_5521766	
query	
value
attention_mask

unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
identity

identity_1¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallqueryvalueattention_maskunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈÈ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_attention_layer_call_and_return_conditional_losses_5520326t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ {

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈ :ÿÿÿÿÿÿÿÿÿÈ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 

_user_specified_namequery:SO
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 

_user_specified_namevalue:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(
_user_specified_nameattention_mask


c
D__inference_dropout_layer_call_and_return_conditional_losses_5522063

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs

b
)__inference_dropout_layer_call_fn_5522046

inputs
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_5520662t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
î

,__inference_sequential_layer_call_fn_5520190
dense_input
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_5520174t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
%
_user_specified_namedense_input
ô	
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_5520596

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬(
Ç
G__inference_sequential_layer_call_and_return_conditional_losses_5521937

inputs9
'dense_tensordot_readvariableop_resource:  3
%dense_biasadd_readvariableop_resource: 
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢.dense/kernel/Regularizer/Square/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: _
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ a

dense/ReluReludense/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype0
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: l
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ ·
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ 
 
_user_specified_nameinputs
³
Î4
#__inference__traced_restore_5522705
file_prefixG
3assignvariableop_review_classifier_t_emb_embeddings:
  H
5assignvariableop_1_review_classifier_p_emb_embeddings:	È Q
;assignvariableop_2_review_classifier_attention_query_kernel:  K
9assignvariableop_3_review_classifier_attention_query_bias: O
9assignvariableop_4_review_classifier_attention_key_kernel:  I
7assignvariableop_5_review_classifier_attention_key_bias: Q
;assignvariableop_6_review_classifier_attention_value_kernel:  K
9assignvariableop_7_review_classifier_attention_value_bias: \
Fassignvariableop_8_review_classifier_attention_attention_output_kernel:  R
Dassignvariableop_9_review_classifier_attention_attention_output_bias: 2
 assignvariableop_10_dense_kernel:  ,
assignvariableop_11_dense_bias: M
?assignvariableop_12_review_classifier_layer_normalization_gamma: L
>assignvariableop_13_review_classifier_layer_normalization_beta: O
Aassignvariableop_14_review_classifier_layer_normalization_1_gamma: N
@assignvariableop_15_review_classifier_layer_normalization_1_beta: F
4assignvariableop_16_review_classifier_dense_1_kernel: @
2assignvariableop_17_review_classifier_dense_1_bias:F
4assignvariableop_18_review_classifier_dense_2_kernel:@
2assignvariableop_19_review_classifier_dense_2_bias:'
assignvariableop_20_adam_iter:	 )
assignvariableop_21_adam_beta_1: )
assignvariableop_22_adam_beta_2: (
assignvariableop_23_adam_decay: 0
&assignvariableop_24_adam_learning_rate: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: #
assignvariableop_27_total: #
assignvariableop_28_count: Q
=assignvariableop_29_adam_review_classifier_t_emb_embeddings_m:
  P
=assignvariableop_30_adam_review_classifier_p_emb_embeddings_m:	È Y
Cassignvariableop_31_adam_review_classifier_attention_query_kernel_m:  S
Aassignvariableop_32_adam_review_classifier_attention_query_bias_m: W
Aassignvariableop_33_adam_review_classifier_attention_key_kernel_m:  Q
?assignvariableop_34_adam_review_classifier_attention_key_bias_m: Y
Cassignvariableop_35_adam_review_classifier_attention_value_kernel_m:  S
Aassignvariableop_36_adam_review_classifier_attention_value_bias_m: d
Nassignvariableop_37_adam_review_classifier_attention_attention_output_kernel_m:  Z
Lassignvariableop_38_adam_review_classifier_attention_attention_output_bias_m: 9
'assignvariableop_39_adam_dense_kernel_m:  3
%assignvariableop_40_adam_dense_bias_m: T
Fassignvariableop_41_adam_review_classifier_layer_normalization_gamma_m: S
Eassignvariableop_42_adam_review_classifier_layer_normalization_beta_m: V
Hassignvariableop_43_adam_review_classifier_layer_normalization_1_gamma_m: U
Gassignvariableop_44_adam_review_classifier_layer_normalization_1_beta_m: M
;assignvariableop_45_adam_review_classifier_dense_1_kernel_m: G
9assignvariableop_46_adam_review_classifier_dense_1_bias_m:M
;assignvariableop_47_adam_review_classifier_dense_2_kernel_m:G
9assignvariableop_48_adam_review_classifier_dense_2_bias_m:Q
=assignvariableop_49_adam_review_classifier_t_emb_embeddings_v:
  P
=assignvariableop_50_adam_review_classifier_p_emb_embeddings_v:	È Y
Cassignvariableop_51_adam_review_classifier_attention_query_kernel_v:  S
Aassignvariableop_52_adam_review_classifier_attention_query_bias_v: W
Aassignvariableop_53_adam_review_classifier_attention_key_kernel_v:  Q
?assignvariableop_54_adam_review_classifier_attention_key_bias_v: Y
Cassignvariableop_55_adam_review_classifier_attention_value_kernel_v:  S
Aassignvariableop_56_adam_review_classifier_attention_value_bias_v: d
Nassignvariableop_57_adam_review_classifier_attention_attention_output_kernel_v:  Z
Lassignvariableop_58_adam_review_classifier_attention_attention_output_bias_v: 9
'assignvariableop_59_adam_dense_kernel_v:  3
%assignvariableop_60_adam_dense_bias_v: T
Fassignvariableop_61_adam_review_classifier_layer_normalization_gamma_v: S
Eassignvariableop_62_adam_review_classifier_layer_normalization_beta_v: V
Hassignvariableop_63_adam_review_classifier_layer_normalization_1_gamma_v: U
Gassignvariableop_64_adam_review_classifier_layer_normalization_1_beta_v: M
;assignvariableop_65_adam_review_classifier_dense_1_kernel_v: G
9assignvariableop_66_adam_review_classifier_dense_1_bias_v:M
;assignvariableop_67_adam_review_classifier_dense_2_kernel_v:G
9assignvariableop_68_adam_review_classifier_dense_2_bias_v:
identity_70¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¨ 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*Î
valueÄBÁFB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÿ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*¡
valueBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ÿ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*®
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*T
dtypesJ
H2F	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp3assignvariableop_review_classifier_t_emb_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_1AssignVariableOp5assignvariableop_1_review_classifier_p_emb_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_2AssignVariableOp;assignvariableop_2_review_classifier_attention_query_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_3AssignVariableOp9assignvariableop_3_review_classifier_attention_query_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_4AssignVariableOp9assignvariableop_4_review_classifier_attention_key_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_5AssignVariableOp7assignvariableop_5_review_classifier_attention_key_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_6AssignVariableOp;assignvariableop_6_review_classifier_attention_value_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_7AssignVariableOp9assignvariableop_7_review_classifier_attention_value_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_8AssignVariableOpFassignvariableop_8_review_classifier_attention_attention_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_9AssignVariableOpDassignvariableop_9_review_classifier_attention_attention_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_12AssignVariableOp?assignvariableop_12_review_classifier_layer_normalization_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_13AssignVariableOp>assignvariableop_13_review_classifier_layer_normalization_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_14AssignVariableOpAassignvariableop_14_review_classifier_layer_normalization_1_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_15AssignVariableOp@assignvariableop_15_review_classifier_layer_normalization_1_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_16AssignVariableOp4assignvariableop_16_review_classifier_dense_1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_17AssignVariableOp2assignvariableop_17_review_classifier_dense_1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_18AssignVariableOp4assignvariableop_18_review_classifier_dense_2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_19AssignVariableOp2assignvariableop_19_review_classifier_dense_2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_29AssignVariableOp=assignvariableop_29_adam_review_classifier_t_emb_embeddings_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_30AssignVariableOp=assignvariableop_30_adam_review_classifier_p_emb_embeddings_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_31AssignVariableOpCassignvariableop_31_adam_review_classifier_attention_query_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_32AssignVariableOpAassignvariableop_32_adam_review_classifier_attention_query_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_33AssignVariableOpAassignvariableop_33_adam_review_classifier_attention_key_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_34AssignVariableOp?assignvariableop_34_adam_review_classifier_attention_key_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_35AssignVariableOpCassignvariableop_35_adam_review_classifier_attention_value_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_36AssignVariableOpAassignvariableop_36_adam_review_classifier_attention_value_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_37AssignVariableOpNassignvariableop_37_adam_review_classifier_attention_attention_output_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_38AssignVariableOpLassignvariableop_38_adam_review_classifier_attention_attention_output_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_dense_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_41AssignVariableOpFassignvariableop_41_adam_review_classifier_layer_normalization_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_42AssignVariableOpEassignvariableop_42_adam_review_classifier_layer_normalization_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_43AssignVariableOpHassignvariableop_43_adam_review_classifier_layer_normalization_1_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_44AssignVariableOpGassignvariableop_44_adam_review_classifier_layer_normalization_1_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_45AssignVariableOp;assignvariableop_45_adam_review_classifier_dense_1_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_46AssignVariableOp9assignvariableop_46_adam_review_classifier_dense_1_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_47AssignVariableOp;assignvariableop_47_adam_review_classifier_dense_2_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_48AssignVariableOp9assignvariableop_48_adam_review_classifier_dense_2_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_49AssignVariableOp=assignvariableop_49_adam_review_classifier_t_emb_embeddings_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_50AssignVariableOp=assignvariableop_50_adam_review_classifier_p_emb_embeddings_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_51AssignVariableOpCassignvariableop_51_adam_review_classifier_attention_query_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_52AssignVariableOpAassignvariableop_52_adam_review_classifier_attention_query_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_53AssignVariableOpAassignvariableop_53_adam_review_classifier_attention_key_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_54AssignVariableOp?assignvariableop_54_adam_review_classifier_attention_key_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_55AssignVariableOpCassignvariableop_55_adam_review_classifier_attention_value_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_56AssignVariableOpAassignvariableop_56_adam_review_classifier_attention_value_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_57AssignVariableOpNassignvariableop_57_adam_review_classifier_attention_attention_output_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_58AssignVariableOpLassignvariableop_58_adam_review_classifier_attention_attention_output_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOp'assignvariableop_59_adam_dense_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOp%assignvariableop_60_adam_dense_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_61AssignVariableOpFassignvariableop_61_adam_review_classifier_layer_normalization_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_62AssignVariableOpEassignvariableop_62_adam_review_classifier_layer_normalization_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_63AssignVariableOpHassignvariableop_63_adam_review_classifier_layer_normalization_1_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_64AssignVariableOpGassignvariableop_64_adam_review_classifier_layer_normalization_1_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_65AssignVariableOp;assignvariableop_65_adam_review_classifier_dense_1_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_66AssignVariableOp9assignvariableop_66_adam_review_classifier_dense_1_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_67AssignVariableOp;assignvariableop_67_adam_review_classifier_dense_2_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_68AssignVariableOp9assignvariableop_68_adam_review_classifier_dense_2_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ½
Identity_69Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_70IdentityIdentity_69:output:0^NoOp_1*
T0*
_output_shapes
: ª
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_70Identity_70:output:0*¡
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_default
<
input_11
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿÈ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ëÒ

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	token_emb
	pos_emb

att
ffn

layernorm1

layernorm2
dropout1
dropout2
pool
dropout3
	dense
dropout4
	stars
	optimizer

signatures"
_tf_keras_model
¶
0
1
2
3
4
5
6
7
8
 9
!10
"11
#12
$13
%14
&15
'16
(17
)18
*19"
trackable_list_wrapper
¶
0
1
2
3
4
5
6
7
8
 9
!10
"11
#12
$13
%14
&15
'16
(17
)18
*19"
trackable_list_wrapper
'
+0"
trackable_list_wrapper
Ê
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object

1trace_0
2trace_1
3trace_2
4trace_32
3__inference_review_classifier_layer_call_fn_5520533
3__inference_review_classifier_layer_call_fn_5521286
3__inference_review_classifier_layer_call_fn_5521331
3__inference_review_classifier_layer_call_fn_5520998Á
¸²´
FullArgSpec2
args*'
jself
jx

jtraining
j
return_att
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z1trace_0z2trace_1z3trace_2z4trace_3
ï
5trace_0
6trace_1
7trace_2
8trace_32
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521495
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521686
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521087
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521176Á
¸²´
FullArgSpec2
args*'
jself
jx

jtraining
j
return_att
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z5trace_0z6trace_1z7trace_2z8trace_3
ÍBÊ
"__inference__wrapped_model_5520074input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
µ
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer

E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
K_query_dense
L
_key_dense
M_value_dense
N_softmax
O_dropout_layer
P_output_dense"
_tf_keras_layer
Ñ
Qlayer_with_weights-0
Qlayer-0
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_sequential
Ä
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^axis
	#gamma
$beta"
_tf_keras_layer
Ä
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
eaxis
	%gamma
&beta"
_tf_keras_layer
¼
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
l_random_generator"
_tf_keras_layer
¼
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
s_random_generator"
_tf_keras_layer
¥
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
½
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
è
	iter
beta_1
beta_2

decay
learning_ratemëmìmímîmïmðmñmòmó mô!mõ"mö#m÷$mø%mù&mú'mû(mü)mý*mþvÿvvvvvvvv v!v"v#v$v%v&v'v(v)v*v"
	optimizer
-
serving_default"
signature_map
6:4
  2"review_classifier/t_emb/embeddings
5:3	È 2"review_classifier/p_emb/embeddings
>:<  2(review_classifier/attention/query/kernel
8:6 2&review_classifier/attention/query/bias
<::  2&review_classifier/attention/key/kernel
6:4 2$review_classifier/attention/key/bias
>:<  2(review_classifier/attention/value/kernel
8:6 2&review_classifier/attention/value/bias
I:G  23review_classifier/attention/attention_output/kernel
?:= 21review_classifier/attention/attention_output/bias
:  2dense/kernel
: 2
dense/bias
9:7 2+review_classifier/layer_normalization/gamma
8:6 2*review_classifier/layer_normalization/beta
;:9 2-review_classifier/layer_normalization_1/gamma
::8 2,review_classifier/layer_normalization_1/beta
2:0 2 review_classifier/dense_1/kernel
,:*2review_classifier/dense_1/bias
2:02 review_classifier/dense_2/kernel
,:*2review_classifier/dense_2/bias
Ð
trace_02±
__inference_loss_fn_0_5521709
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ ztrace_0
 "
trackable_list_wrapper
~
0
	1

2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
3__inference_review_classifier_layer_call_fn_5520533input_1"Á
¸²´
FullArgSpec2
args*'
jself
jx

jtraining
j
return_att
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bþ
3__inference_review_classifier_layer_call_fn_5521286x"Á
¸²´
FullArgSpec2
args*'
jself
jx

jtraining
j
return_att
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bþ
3__inference_review_classifier_layer_call_fn_5521331x"Á
¸²´
FullArgSpec2
args*'
jself
jx

jtraining
j
return_att
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
3__inference_review_classifier_layer_call_fn_5520998input_1"Á
¸²´
FullArgSpec2
args*'
jself
jx

jtraining
j
return_att
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521495x"Á
¸²´
FullArgSpec2
args*'
jself
jx

jtraining
j
return_att
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521686x"Á
¸²´
FullArgSpec2
args*'
jself
jx

jtraining
j
return_att
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢B
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521087input_1"Á
¸²´
FullArgSpec2
args*'
jself
jx

jtraining
j
return_att
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢B
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521176input_1"Á
¸²´
FullArgSpec2
args*'
jself
jx

jtraining
j
return_att
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
í
¢trace_02Î
'__inference_t_emb_layer_call_fn_5521716¢
²
FullArgSpec
args
jself
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
annotationsª *
 z¢trace_0

£trace_02é
B__inference_t_emb_layer_call_and_return_conditional_losses_5521725¢
²
FullArgSpec
args
jself
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
annotationsª *
 z£trace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
í
©trace_02Î
'__inference_p_emb_layer_call_fn_5521732¢
²
FullArgSpec
args
jself
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
annotationsª *
 z©trace_0

ªtrace_02é
B__inference_p_emb_layer_call_and_return_conditional_losses_5521741¢
²
FullArgSpec
args
jself
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
annotationsª *
 zªtrace_0
X
0
1
2
3
4
5
6
 7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
 7"
trackable_list_wrapper
 "
trackable_list_wrapper
²
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object

°trace_0
±trace_12Ù
+__inference_attention_layer_call_fn_5521766
+__inference_attention_layer_call_fn_5521791ü
ó²ï
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z°trace_0z±trace_1
Ê
²trace_0
³trace_12
F__inference_attention_layer_call_and_return_conditional_losses_5521834
F__inference_attention_layer_call_and_return_conditional_losses_5521876ü
ó²ï
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z²trace_0z³trace_1
ô
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses
ºpartial_output_shape
»full_output_shape

kernel
bias"
_tf_keras_layer
ô
¼	variables
½trainable_variables
¾regularization_losses
¿	keras_api
À__call__
+Á&call_and_return_all_conditional_losses
Âpartial_output_shape
Ãfull_output_shape

kernel
bias"
_tf_keras_layer
ô
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses
Êpartial_output_shape
Ëfull_output_shape

kernel
bias"
_tf_keras_layer
«
Ì	variables
Ítrainable_variables
Îregularization_losses
Ï	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ò	variables
Ótrainable_variables
Ôregularization_losses
Õ	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses
Ø_random_generator"
_tf_keras_layer
ô
Ù	variables
Útrainable_variables
Ûregularization_losses
Ü	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses
ßpartial_output_shape
àfull_output_shape

kernel
 bias"
_tf_keras_layer
Á
á	variables
âtrainable_variables
ãregularization_losses
ä	keras_api
å__call__
+æ&call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
(
ç0"
trackable_list_wrapper
²
ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
î
ítrace_0
îtrace_1
ïtrace_2
ðtrace_32û
,__inference_sequential_layer_call_fn_5520138
,__inference_sequential_layer_call_fn_5521891
,__inference_sequential_layer_call_fn_5521900
,__inference_sequential_layer_call_fn_5520190À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zítrace_0zîtrace_1zïtrace_2zðtrace_3
Ú
ñtrace_0
òtrace_1
ótrace_2
ôtrace_32ç
G__inference_sequential_layer_call_and_return_conditional_losses_5521937
G__inference_sequential_layer_call_and_return_conditional_losses_5521974
G__inference_sequential_layer_call_and_return_conditional_losses_5520205
G__inference_sequential_layer_call_and_return_conditional_losses_5520220À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zñtrace_0zòtrace_1zótrace_2zôtrace_3
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
û
útrace_02Ü
5__inference_layer_normalization_layer_call_fn_5521983¢
²
FullArgSpec
args
jself
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
annotationsª *
 zútrace_0

ûtrace_02÷
P__inference_layer_normalization_layer_call_and_return_conditional_losses_5522005¢
²
FullArgSpec
args
jself
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
annotationsª *
 zûtrace_0
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
ý
trace_02Þ
7__inference_layer_normalization_1_layer_call_fn_5522014¢
²
FullArgSpec
args
jself
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
annotationsª *
 ztrace_0

trace_02ù
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5522036¢
²
FullArgSpec
args
jself
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
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
È
trace_0
trace_12
)__inference_dropout_layer_call_fn_5522041
)__inference_dropout_layer_call_fn_5522046´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
þ
trace_0
trace_12Ã
D__inference_dropout_layer_call_and_return_conditional_losses_5522051
D__inference_dropout_layer_call_and_return_conditional_losses_5522063´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
Ì
trace_0
trace_12
+__inference_dropout_1_layer_call_fn_5522068
+__inference_dropout_1_layer_call_fn_5522073´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ç
F__inference_dropout_1_layer_call_and_return_conditional_losses_5522078
F__inference_dropout_1_layer_call_and_return_conditional_losses_5522090´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object

trace_02î
:__inference_global_average_pooling1d_layer_call_fn_5522095¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
¨
trace_02
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5522101¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ì
¡trace_0
¢trace_12
+__inference_dropout_2_layer_call_fn_5522106
+__inference_dropout_2_layer_call_fn_5522111´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¡trace_0z¢trace_1

£trace_0
¤trace_12Ç
F__inference_dropout_2_layer_call_and_return_conditional_losses_5522116
F__inference_dropout_2_layer_call_and_return_conditional_losses_5522128´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z£trace_0z¤trace_1
"
_generic_user_object
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
'
+0"
trackable_list_wrapper
¸
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ï
ªtrace_02Ð
)__inference_dense_1_layer_call_fn_5522137¢
²
FullArgSpec
args
jself
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
annotationsª *
 zªtrace_0

«trace_02ë
D__inference_dense_1_layer_call_and_return_conditional_losses_5522154¢
²
FullArgSpec
args
jself
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
annotationsª *
 z«trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ì
±trace_0
²trace_12
+__inference_dropout_3_layer_call_fn_5522159
+__inference_dropout_3_layer_call_fn_5522164´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z±trace_0z²trace_1

³trace_0
´trace_12Ç
F__inference_dropout_3_layer_call_and_return_conditional_losses_5522169
F__inference_dropout_3_layer_call_and_return_conditional_losses_5522181´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z³trace_0z´trace_1
"
_generic_user_object
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ï
ºtrace_02Ð
)__inference_dense_2_layer_call_fn_5522190¢
²
FullArgSpec
args
jself
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
annotationsª *
 zºtrace_0

»trace_02ë
D__inference_dense_2_layer_call_and_return_conditional_losses_5522201¢
²
FullArgSpec
args
jself
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
annotationsª *
 z»trace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÌBÉ
%__inference_signature_wrapper_5521241input_1"
²
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
annotationsª *
 
´B±
__inference_loss_fn_0_5521709"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
R
¼	variables
½	keras_api

¾total

¿count"
_tf_keras_metric
c
À	variables
Á	keras_api

Âtotal

Ãcount
Ä
_fn_kwargs"
_tf_keras_metric
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
ÛBØ
'__inference_t_emb_layer_call_fn_5521716inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
öBó
B__inference_t_emb_layer_call_and_return_conditional_losses_5521725inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
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
ÛBØ
'__inference_p_emb_layer_call_fn_5521732inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
öBó
B__inference_p_emb_layer_call_and_return_conditional_losses_5521741inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
 "
trackable_list_wrapper
J
K0
L1
M2
N3
O4
P5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÏBÌ
+__inference_attention_layer_call_fn_5521766queryvalueattention_mask"ü
ó²ï
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÏBÌ
+__inference_attention_layer_call_fn_5521791queryvalueattention_mask"ü
ó²ï
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
êBç
F__inference_attention_layer_call_and_return_conditional_losses_5521834queryvalueattention_mask"ü
ó²ï
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
êBç
F__inference_attention_layer_call_and_return_conditional_losses_5521876queryvalueattention_mask"ü
ó²ï
FullArgSpece
args]Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
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
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
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
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
¼	variables
½trainable_variables
¾regularization_losses
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
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
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
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
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
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
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
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
annotationsª *
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
trackable_list_wrapper
¸
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
Ì	variables
Ítrainable_variables
Îregularization_losses
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
Ò	variables
Ótrainable_variables
Ôregularization_losses
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
"
_generic_user_object
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
¸
Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
Ù	variables
Útrainable_variables
Ûregularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
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
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
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
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
(
ç0"
trackable_list_wrapper
¸
ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
á	variables
âtrainable_variables
ãregularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
í
ètrace_02Î
'__inference_dense_layer_call_fn_5522210¢
²
FullArgSpec
args
jself
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
annotationsª *
 zètrace_0

étrace_02é
B__inference_dense_layer_call_and_return_conditional_losses_5522247¢
²
FullArgSpec
args
jself
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
annotationsª *
 zétrace_0
Ð
êtrace_02±
__inference_loss_fn_1_5522258
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ zêtrace_0
 "
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_sequential_layer_call_fn_5520138dense_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þBû
,__inference_sequential_layer_call_fn_5521891inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þBû
,__inference_sequential_layer_call_fn_5521900inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
,__inference_sequential_layer_call_fn_5520190dense_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_5521937inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_5521974inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_5520205dense_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
G__inference_sequential_layer_call_and_return_conditional_losses_5520220dense_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
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
éBæ
5__inference_layer_normalization_layer_call_fn_5521983inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
B
P__inference_layer_normalization_layer_call_and_return_conditional_losses_5522005inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
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
ëBè
7__inference_layer_normalization_1_layer_call_fn_5522014inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
B
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5522036inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
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
ïBì
)__inference_dropout_layer_call_fn_5522041inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ïBì
)__inference_dropout_layer_call_fn_5522046inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
D__inference_dropout_layer_call_and_return_conditional_losses_5522051inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
D__inference_dropout_layer_call_and_return_conditional_losses_5522063inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
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
ñBî
+__inference_dropout_1_layer_call_fn_5522068inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ñBî
+__inference_dropout_1_layer_call_fn_5522073inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_dropout_1_layer_call_and_return_conditional_losses_5522078inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_dropout_1_layer_call_and_return_conditional_losses_5522090inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
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
ûBø
:__inference_global_average_pooling1d_layer_call_fn_5522095inputs"¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5522101inputs"¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ñBî
+__inference_dropout_2_layer_call_fn_5522106inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ñBî
+__inference_dropout_2_layer_call_fn_5522111inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_dropout_2_layer_call_and_return_conditional_losses_5522116inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_dropout_2_layer_call_and_return_conditional_losses_5522128inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_1_layer_call_fn_5522137inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
øBõ
D__inference_dense_1_layer_call_and_return_conditional_losses_5522154inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
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
ñBî
+__inference_dropout_3_layer_call_fn_5522159inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ñBî
+__inference_dropout_3_layer_call_fn_5522164inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_dropout_3_layer_call_and_return_conditional_losses_5522169inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_dropout_3_layer_call_and_return_conditional_losses_5522181inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
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
ÝBÚ
)__inference_dense_2_layer_call_fn_5522190inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
øBõ
D__inference_dense_2_layer_call_and_return_conditional_losses_5522201inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
0
¾0
¿1"
trackable_list_wrapper
.
¼	variables"
_generic_user_object
:  (2total
:  (2count
0
Â0
Ã1"
trackable_list_wrapper
.
À	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ç0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
'__inference_dense_layer_call_fn_5522210inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
öBó
B__inference_dense_layer_call_and_return_conditional_losses_5522247inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
´B±
__inference_loss_fn_1_5522258"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
;:9
  2)Adam/review_classifier/t_emb/embeddings/m
::8	È 2)Adam/review_classifier/p_emb/embeddings/m
C:A  2/Adam/review_classifier/attention/query/kernel/m
=:; 2-Adam/review_classifier/attention/query/bias/m
A:?  2-Adam/review_classifier/attention/key/kernel/m
;:9 2+Adam/review_classifier/attention/key/bias/m
C:A  2/Adam/review_classifier/attention/value/kernel/m
=:; 2-Adam/review_classifier/attention/value/bias/m
N:L  2:Adam/review_classifier/attention/attention_output/kernel/m
D:B 28Adam/review_classifier/attention/attention_output/bias/m
#:!  2Adam/dense/kernel/m
: 2Adam/dense/bias/m
>:< 22Adam/review_classifier/layer_normalization/gamma/m
=:; 21Adam/review_classifier/layer_normalization/beta/m
@:> 24Adam/review_classifier/layer_normalization_1/gamma/m
?:= 23Adam/review_classifier/layer_normalization_1/beta/m
7:5 2'Adam/review_classifier/dense_1/kernel/m
1:/2%Adam/review_classifier/dense_1/bias/m
7:52'Adam/review_classifier/dense_2/kernel/m
1:/2%Adam/review_classifier/dense_2/bias/m
;:9
  2)Adam/review_classifier/t_emb/embeddings/v
::8	È 2)Adam/review_classifier/p_emb/embeddings/v
C:A  2/Adam/review_classifier/attention/query/kernel/v
=:; 2-Adam/review_classifier/attention/query/bias/v
A:?  2-Adam/review_classifier/attention/key/kernel/v
;:9 2+Adam/review_classifier/attention/key/bias/v
C:A  2/Adam/review_classifier/attention/value/kernel/v
=:; 2-Adam/review_classifier/attention/value/bias/v
N:L  2:Adam/review_classifier/attention/attention_output/kernel/v
D:B 28Adam/review_classifier/attention/attention_output/bias/v
#:!  2Adam/dense/kernel/v
: 2Adam/dense/bias/v
>:< 22Adam/review_classifier/layer_normalization/gamma/v
=:; 21Adam/review_classifier/layer_normalization/beta/v
@:> 24Adam/review_classifier/layer_normalization_1/gamma/v
?:= 23Adam/review_classifier/layer_normalization_1/beta/v
7:5 2'Adam/review_classifier/dense_1/kernel/v
1:/2%Adam/review_classifier/dense_1/bias/v
7:52'Adam/review_classifier/dense_2/kernel/v
1:/2%Adam/review_classifier/dense_2/bias/v¤
"__inference__wrapped_model_5520074~ #$!"%&'()*1¢.
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿÈ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿÏ
F__inference_attention_layer_call_and_return_conditional_losses_5521834 ¢
¢
$!
queryÿÿÿÿÿÿÿÿÿÈ 
$!
valueÿÿÿÿÿÿÿÿÿÈ 

 
1.
attention_maskÿÿÿÿÿÿÿÿÿÈ

p
p 
ª "Z¢W
P¢M
"
0/0ÿÿÿÿÿÿÿÿÿÈ 
'$
0/1ÿÿÿÿÿÿÿÿÿÈÈ
 Ï
F__inference_attention_layer_call_and_return_conditional_losses_5521876 ¢
¢
$!
queryÿÿÿÿÿÿÿÿÿÈ 
$!
valueÿÿÿÿÿÿÿÿÿÈ 

 
1.
attention_maskÿÿÿÿÿÿÿÿÿÈ

p
p
ª "Z¢W
P¢M
"
0/0ÿÿÿÿÿÿÿÿÿÈ 
'$
0/1ÿÿÿÿÿÿÿÿÿÈÈ
 ¦
+__inference_attention_layer_call_fn_5521766ö ¢
¢
$!
queryÿÿÿÿÿÿÿÿÿÈ 
$!
valueÿÿÿÿÿÿÿÿÿÈ 

 
1.
attention_maskÿÿÿÿÿÿÿÿÿÈ

p
p 
ª "L¢I
 
0ÿÿÿÿÿÿÿÿÿÈ 
%"
1ÿÿÿÿÿÿÿÿÿÈÈ¦
+__inference_attention_layer_call_fn_5521791ö ¢
¢
$!
queryÿÿÿÿÿÿÿÿÿÈ 
$!
valueÿÿÿÿÿÿÿÿÿÈ 

 
1.
attention_maskÿÿÿÿÿÿÿÿÿÈ

p
p
ª "L¢I
 
0ÿÿÿÿÿÿÿÿÿÈ 
%"
1ÿÿÿÿÿÿÿÿÿÈÈ¤
D__inference_dense_1_layer_call_and_return_conditional_losses_5522154\'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_1_layer_call_fn_5522137O'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_2_layer_call_and_return_conditional_losses_5522201\)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_2_layer_call_fn_5522190O)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
B__inference_dense_layer_call_and_return_conditional_losses_5522247f!"4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÈ 
 
'__inference_dense_layer_call_fn_5522210Y!"4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
ª "ÿÿÿÿÿÿÿÿÿÈ °
F__inference_dropout_1_layer_call_and_return_conditional_losses_5522078f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÈ 
 °
F__inference_dropout_1_layer_call_and_return_conditional_losses_5522090f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÈ 
 
+__inference_dropout_1_layer_call_fn_5522068Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
p 
ª "ÿÿÿÿÿÿÿÿÿÈ 
+__inference_dropout_1_layer_call_fn_5522073Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
p
ª "ÿÿÿÿÿÿÿÿÿÈ ¦
F__inference_dropout_2_layer_call_and_return_conditional_losses_5522116\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ¦
F__inference_dropout_2_layer_call_and_return_conditional_losses_5522128\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ~
+__inference_dropout_2_layer_call_fn_5522106O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ ~
+__inference_dropout_2_layer_call_fn_5522111O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ ¦
F__inference_dropout_3_layer_call_and_return_conditional_losses_5522169\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¦
F__inference_dropout_3_layer_call_and_return_conditional_losses_5522181\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dropout_3_layer_call_fn_5522159O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ~
+__inference_dropout_3_layer_call_fn_5522164O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ®
D__inference_dropout_layer_call_and_return_conditional_losses_5522051f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÈ 
 ®
D__inference_dropout_layer_call_and_return_conditional_losses_5522063f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÈ 
 
)__inference_dropout_layer_call_fn_5522041Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
p 
ª "ÿÿÿÿÿÿÿÿÿÈ 
)__inference_dropout_layer_call_fn_5522046Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
p
ª "ÿÿÿÿÿÿÿÿÿÈ Ô
U__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5522101{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¬
:__inference_global_average_pooling1d_layer_call_fn_5522095nI¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
R__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5522036f%&4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÈ 
 
7__inference_layer_normalization_1_layer_call_fn_5522014Y%&4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
ª "ÿÿÿÿÿÿÿÿÿÈ º
P__inference_layer_normalization_layer_call_and_return_conditional_losses_5522005f#$4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÈ 
 
5__inference_layer_normalization_layer_call_fn_5521983Y#$4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
ª "ÿÿÿÿÿÿÿÿÿÈ <
__inference_loss_fn_0_5521709'¢

¢ 
ª " <
__inference_loss_fn_1_5522258!¢

¢ 
ª " 
B__inference_p_emb_layer_call_and_return_conditional_losses_5521741G#¢ 
¢

inputsÈ
ª "¢

0	È 
 e
'__inference_p_emb_layer_call_fn_5521732:#¢ 
¢

inputsÈ
ª "	È Ê
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521087x #$!"%&'()*9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿÈ
p 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ê
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521176x #$!"%&'()*9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿÈ
p
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521495r #$!"%&'()*3¢0
)¢&

xÿÿÿÿÿÿÿÿÿÈ
p 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
N__inference_review_classifier_layer_call_and_return_conditional_losses_5521686r #$!"%&'()*3¢0
)¢&

xÿÿÿÿÿÿÿÿÿÈ
p
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¢
3__inference_review_classifier_layer_call_fn_5520533k #$!"%&'()*9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿÈ
p 
p 
ª "ÿÿÿÿÿÿÿÿÿ¢
3__inference_review_classifier_layer_call_fn_5520998k #$!"%&'()*9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿÈ
p
p 
ª "ÿÿÿÿÿÿÿÿÿ
3__inference_review_classifier_layer_call_fn_5521286e #$!"%&'()*3¢0
)¢&

xÿÿÿÿÿÿÿÿÿÈ
p 
p 
ª "ÿÿÿÿÿÿÿÿÿ
3__inference_review_classifier_layer_call_fn_5521331e #$!"%&'()*3¢0
)¢&

xÿÿÿÿÿÿÿÿÿÈ
p
p 
ª "ÿÿÿÿÿÿÿÿÿ¾
G__inference_sequential_layer_call_and_return_conditional_losses_5520205s!"A¢>
7¢4
*'
dense_inputÿÿÿÿÿÿÿÿÿÈ 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÈ 
 ¾
G__inference_sequential_layer_call_and_return_conditional_losses_5520220s!"A¢>
7¢4
*'
dense_inputÿÿÿÿÿÿÿÿÿÈ 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÈ 
 ¹
G__inference_sequential_layer_call_and_return_conditional_losses_5521937n!"<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÈ 
 ¹
G__inference_sequential_layer_call_and_return_conditional_losses_5521974n!"<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÈ 
 
,__inference_sequential_layer_call_fn_5520138f!"A¢>
7¢4
*'
dense_inputÿÿÿÿÿÿÿÿÿÈ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿÈ 
,__inference_sequential_layer_call_fn_5520190f!"A¢>
7¢4
*'
dense_inputÿÿÿÿÿÿÿÿÿÈ 
p

 
ª "ÿÿÿÿÿÿÿÿÿÈ 
,__inference_sequential_layer_call_fn_5521891a!"<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿÈ 
,__inference_sequential_layer_call_fn_5521900a!"<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿÈ 
p

 
ª "ÿÿÿÿÿÿÿÿÿÈ ³
%__inference_signature_wrapper_5521241 #$!"%&'()*<¢9
¢ 
2ª/
-
input_1"
input_1ÿÿÿÿÿÿÿÿÿÈ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ§
B__inference_t_emb_layer_call_and_return_conditional_losses_5521725a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÈ 
 
'__inference_t_emb_layer_call_fn_5521716T0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿÈ 