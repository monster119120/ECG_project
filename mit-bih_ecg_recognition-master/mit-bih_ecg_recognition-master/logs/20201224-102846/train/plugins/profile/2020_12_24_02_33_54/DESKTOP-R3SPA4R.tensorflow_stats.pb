"�g
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
BHostIDLE"IDLE1���̌E�@A���̌E�@a@�:h��?i@�:h��?�Unknown
lHostConv2D"sequential/conv1d/conv1d(1     $�@9     $�@A     $�@I     $�@a7�V[�?in]����?�Unknown
uHostMaxPool" sequential/max_pooling1d/MaxPool(133333��@933333��@A33333��@I33333��@am����?i�H�/?�?�Unknown
�HostConv2DBackpropFilter";gradient_tape/sequential/conv1d/conv1d/Conv2DBackpropFilter(1     H�@9     H�@A     H�@I     H�@a� ��/�?i͊')"?�?�Unknown
�HostMaxPoolGrad"<gradient_tape/sequential/max_pooling1d_1/MaxPool/MaxPoolGrad(1����̤�@9����̤�@A����̤�@I����̤�@a�?i��I2-�?�Unknown
�HostConv2DBackpropFilter"=gradient_tape/sequential/conv1d_1/conv1d/Conv2DBackpropFilter(1     ��@9     ��@A     ��@I     ��@aio;�x�?i ����?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1�����M�@9�����M�@A�����M�@I�����M�@a�nǧ;ʗ?it1����?�Unknown�
�	HostConv2DBackpropInput"<gradient_tape/sequential/conv1d_1/conv1d/Conv2DBackpropInput(133333'�@933333'�@A33333'�@I33333'�@a]m��v5�?i߼_��t�?�Unknown
�
HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     �|@9     �|@A     �|@I     �|@a�����?i�j����?�Unknown
nHostConv2D"sequential/conv1d_1/conv1d(1fffff{@9fffff{@Afffff{@Ifffff{@a��R	��?i��D��?�Unknown
�HostMaxPoolGrad":gradient_tape/sequential/max_pooling1d/MaxPool/MaxPoolGrad(1�����<n@9�����<n@A�����<n@I�����<n@a((�XdɄ?i�7vbj��?�Unknown
�HostBiasAddGrad"3gradient_tape/sequential/conv1d/BiasAdd/BiasAddGrad(133333l@933333l@A33333l@I33333l@a�,�n�A�?iT�1�qG�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(133333Ch@933333Ch@A33333Ch@I33333Ch@ayc߭�?i���))��?�Unknown
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1�����h@9�����h@A�����h@I�����h@aE	W�y��?i�P�w��?�Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(133333�g@933333�g@A33333�g@I33333�g@at�T�J�?iul���?�Unknown
nHostBiasAdd"sequential/conv1d/BiasAdd(1333333g@9333333g@A333333g@I333333g@a��b��?i42P!nM�?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1�����ya@9�����ya@A�����ya@I�����ya@a�2�w�x?i��?�{}�?�Unknown
wHostMaxPool""sequential/max_pooling1d_1/MaxPool(133333^@933333^@A33333^@I33333^@a���Nˬt?i'��,զ�?�Unknown
pHostBiasAdd"sequential/conv1d_1/BiasAdd(1fffff�]@9fffff�]@Afffff�]@Ifffff�]@aE�h��wt?i�i�+���?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1     `\@9     `\@A     `\@I     `\@a[����s?im��g���?�Unknown
�HostReluGrad"*gradient_tape/sequential/conv1d_1/ReluGrad(1�����LW@9�����LW@A�����LW@I�����LW@a�!�O|p?i�G�`��?�Unknown
^HostGatherV2"GatherV2(1333333W@9333333W@A333333W@I333333W@a��b��o?i���&�6�?�Unknown
�HostBiasAddGrad"5gradient_tape/sequential/conv1d_1/BiasAdd/BiasAddGrad(1fffff�R@9fffff�R@Afffff�R@Ifffff�R@a��J�Lxi?ia�ws/P�?�Unknown
~HostReluGrad"(gradient_tape/sequential/conv1d/ReluGrad(1�����YI@9�����YI@A�����YI@I�����YI@a�e�Bma?iǼ��a�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(133333sH@933333sH@A33333sH@I33333sH@a{"G���`?i�=�kr�?�Unknown
jHostRelu"sequential/conv1d_1/Relu(1fffff�E@9fffff�E@Afffff�E@Ifffff�E@aJM797�]?i��ٯM��?�Unknown
hHostRelu"sequential/conv1d/Relu(1������=@9������=@A������=@I������=@a�\��e|T?i>y�⋋�?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(133333�<@933333�<@A33333�<@I33333�<@aM��кS?i��Ji��?�Unknown
oHostSoftmax"sequential/dense_1/Softmax(1      4@9      4@A      4@I      4@a�V��uK?i��C(I��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1������3@9������3@A      0@I      0@aѫ^u��E?iZ��ɡ�?�Unknown
d HostDataset"Iterator::Model(1fffff�4@9fffff�4@A333333/@I333333/@a�	l�rE?i��{�%��?�Unknown
`!HostGatherV2"
GatherV2_1(1������,@9������,@A������,@I������,@a*��6�C?ib\/��?�Unknown
i"HostWriteSummary"WriteSummary(1������&@9������&@A������&@I������&@a�b�??i�H_��?�Unknown�
{#HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1������%@9������%@A������%@I������%@a�c���=?i��p���?�Unknown
}$HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      $@9      $@A      $@I      $@a�V��u;?ij6+n!��?�Unknown
t%Host_FusedMatMul"sequential/dense_1/BiasAdd(1������#@9������#@A������#@I������#@a:��9;?i��D����?�Unknown
�&HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1������#@9������#@A������#@I������#@a��`ɫ�:?i�#����?�Unknown
q'HostCast"sequential/dropout/dropout/Cast(1      !@9      !@A      !@I      !@a�����_7?i������?�Unknown
�(HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1������ @9������ @A������ @I������ @a�	�%7?i»�����?�Unknown
�)HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1       @9       @A       @I       @aѫ^u��5?i�g��u��?�Unknown
Z*HostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a���b��4?i�����?�Unknown
�+HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      @9      @A      @I      @a�^��4?i �����?�Unknown
�,HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1ffffff@9ffffff@Affffff@Iffffff@a�]K�3?i��0a��?�Unknown
�-HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1������@9������@A������@I������@a�|�8pl2?i$8�d��?�Unknown
g.HostStridedSlice"strided_slice(1      @9      @A      @I      @a��\/��1?i������?�Unknown
�/HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1ffffff"@9ffffff"@A������@I������@a#�*A�1?iC#���?�Unknown
�0HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      @9      @A      @I      @a� �0?i�C����?�Unknown
�1HostResourceApplyAdam"$Adam/Adam/update_7/ResourceApplyAdam(1������@9������@A������@I������@aQ8\H90?il��*���?�Unknown
�2HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a��b��/?i��*����?�Unknown
s3HostDataset"Iterator::Model::ParallelMapV2(1333333@9333333@A333333@I333333@a
���%-?iqi���?�Unknown
�4HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1      @9      @A      @I      @a�V��u+?ip���s��?�Unknown
�5HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1333333@9333333@A333333@I333333@a__��'?if��:���?�Unknown
V6HostSum"Sum_2(1������@9������@A������@I������@a�	�%'?i3�_��?�Unknown
x7HostDataset"#Iterator::Model::ParallelMapV2::Zip(1�����LA@9�����LA@Affffff@Iffffff@a�<�~[�&?iGВ���?�Unknown
�8HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?apgPi�#?i�eY��?�Unknown
�9HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@apgPi�#?iS�B��?�Unknown
v:HostCast"$sparse_categorical_crossentropy/Cast(1333333@9333333@A333333@I333333@a?E]=ղ"?i'�MMm��?�Unknown
l;HostIteratorGetNext"IteratorGetNext(1ffffff
@9ffffff
@Affffff
@Iffffff
@a&�4&"?i�5����?�Unknown
�<HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1ffffff
@9ffffff
@Affffff
@Iffffff
@a&�4&"?iv����?�Unknown
�=HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1������	@9������	@A������	@I������	@a#�*A�!?i?!Ǣ���?�Unknown
e>Host
LogicalAnd"
LogicalAnd(1333333@9333333@A333333@I333333@a��b��?iV�����?�Unknown�
�?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@aX��
2�?ib�2���?�Unknown
Y@HostPow"Adam/Pow(1������@9������@A������@I������@a(����?iy"xǮ��?�Unknown
�AHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1������@9������@A������@I������@a(����?i��g\���?�Unknown
oBHostMul"sequential/dropout/dropout/Mul(1      @9      @A      @I      @a�V��u?i�xXx��?�Unknown
[CHostAddV2"Adam/add(1������@9������@A������@I������@a4𴚹2?i9N��9��?�Unknown
tDHostAssignAddVariableOp"AssignAddVariableOp(1������@9������@A������@I������@a4𴚹2?i�#�����?�Unknown
bEHostDivNoNan"div_no_nan_1(1������@9������@A������@I������@a4𴚹2?i��|���?�Unknown
vFHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1������ @9������ @A������ @I������ @a�	�%?i�9��u��?�Unknown
[GHostPow"
Adam/Pow_1(1������ @9������ @A������ @I������ @a�	�%?i%zի.��?�Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_2(1������ @9������ @A������ @I������ @a�	�%?is�u���?�Unknown
XIHostEqual"Equal(1       @9       @A       @I       @aѫ^u��?ihe�q���?�Unknown
XJHostCast"Cast_3(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���b��?i{x�>��?�Unknown
~KHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1�������?9�������?A�������?I�������?apgPi�?iG�����?�Unknown
�LHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������%@9������%@A333333�?I333333�?a?E]=ղ?i1�l�r��?�Unknown
TMHostMul"Mul(1333333�?9333333�?A333333�?I333333�?a?E]=ղ?i�2��?�Unknown
wNHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a?E]=ղ?i��ȝ��?�Unknown
}OHostMul",gradient_tape/sequential/dropout/dropout/Mul(1333333�?9333333�?A333333�?I333333�?a?E]=ղ?i�j_3��?�Unknown
�PHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a#�*A�?i��s)���?�Unknown
QHostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1      �?9      �?A      �?I      �?a� �?i���&D��?�Unknown
�RHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1      �?9      �?A      �?I      �?a� �?i�|E$���?�Unknown
]SHostCast"Adam/Cast_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aX��
2�?iϧUC��?�Unknown
�THostReadVariableOp"4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aX��
2�?i��Յ���?�Unknown
VUHostCast"Cast(1�������?9�������?A�������?I�������?a�xa�	�?i4h��0��?�Unknown
uVHostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?a�xa�	�?i��$N���?�Unknown
�WHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�xa�	�?i@�L���?�Unknown
vXHostAssignAddVariableOp"AssignAddVariableOp_4(1333333�?9333333�?A333333�?I333333�?a�4��e
?im��I��?�Unknown
qYHostMul" sequential/dropout/dropout/Mul_1(1333333�?9333333�?A333333�?I333333�?a�4��e
?i��Z����?�Unknown
�ZHostReadVariableOp"6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp(1�������?9�������?A�������?I�������?a4𴚹2?in�@�I��?�Unknown
X[HostCast"Cast_2(1�������?9�������?A�������?I�������?apgPi�?i�>�ݘ��?�Unknown
t\HostReadVariableOp"Adam/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a#�*A�?iY��B���?�Unknown
v]HostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a#�*A�?i"��%��?�Unknown
v^HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aX��
2��>i��S@c��?�Unknown
X_HostCast"Cast_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?aX��
2��>i ��ؠ��?�Unknown
�`HostReadVariableOp"(sequential/conv1d/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aX��
2��>io�q���?�Unknown
�aHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aX��
2��>i��	��?�Unknown
�bHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1ffffff�?9ffffff�?Affffff�?Iffffff�?aX��
2��>iM��Y��?�Unknown
ocHostReadVariableOp"Adam/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�4��e�>ic�m���?�Unknown
`dHostDivNoNan"
div_no_nan(1333333�?9333333�?A333333�?I333333�?a�4��e�>iy�j9���?�Unknown
yeHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a�4��e�>i�.���?�Unknown
�fHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�4��e�>i����,��?�Unknown
vgHostAssignAddVariableOp"AssignAddVariableOp_1(1      �?9      �?A      �?I      �?aѫ^u���>ib��X��?�Unknown
whHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      �?9      �?A      �?I      �?aѫ^u���>i�7τ��?�Unknown
�iHostReadVariableOp"*sequential/conv1d_1/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?aѫ^u���>iܿZΰ��?�Unknown
�jHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?aѫ^u���>i��}����?�Unknown
akHostIdentity"Identity(1�������?9�������?A�������?I�������?a#�*A��>i�������?�Unknown�*�f
lHostConv2D"sequential/conv1d/conv1d(1     $�@9     $�@A     $�@I     $�@a�g�J䤼?i�g�J䤼?�Unknown
uHostMaxPool" sequential/max_pooling1d/MaxPool(133333��@933333��@A33333��@I33333��@a ��/��?i�Q����?�Unknown
�HostConv2DBackpropFilter";gradient_tape/sequential/conv1d/conv1d/Conv2DBackpropFilter(1     H�@9     H�@A     H�@I     H�@a!s�?i��&���?�Unknown
�HostMaxPoolGrad"<gradient_tape/sequential/max_pooling1d_1/MaxPool/MaxPoolGrad(1����̤�@9����̤�@A����̤�@I����̤�@a/�odF�?i��B.��?�Unknown
�HostConv2DBackpropFilter"=gradient_tape/sequential/conv1d_1/conv1d/Conv2DBackpropFilter(1     ��@9     ��@A     ��@I     ��@ao�&��B�?i��s���?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1�����M�@9�����M�@A�����M�@I�����M�@a�k�* �?i�A�>,�?�Unknown�
�HostConv2DBackpropInput"<gradient_tape/sequential/conv1d_1/conv1d/Conv2DBackpropInput(133333'�@933333'�@A33333'�@I33333'�@a���{��?i����C�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     �|@9     �|@A     �|@I     �|@a���`�:�?i$�����?�Unknown
n	HostConv2D"sequential/conv1d_1/conv1d(1fffff{@9fffff{@Afffff{@Ifffff{@a������?i��p�m�?�Unknown
�
HostMaxPoolGrad":gradient_tape/sequential/max_pooling1d/MaxPool/MaxPoolGrad(1�����<n@9�����<n@A�����<n@I�����<n@a�F踝?ij�E�[�?�Unknown
�HostBiasAddGrad"3gradient_tape/sequential/conv1d/BiasAdd/BiasAddGrad(133333l@933333l@A33333l@I33333l@a
&�4��?iO+I_8�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(133333Ch@933333Ch@A33333Ch@I33333Ch@a?P�^ٗ?iH�CT���?�Unknown
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1�����h@9�����h@A�����h@I�����h@a&����?iRo��?�Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(133333�g@933333�g@A33333�g@I33333�g@aD���K�?iK����n�?�Unknown
nHostBiasAdd"sequential/conv1d/BiasAdd(1333333g@9333333g@A333333g@I333333g@a�΂Ζ?i��X�=%�?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1�����ya@9�����ya@A�����ya@I�����ya@a	/�>`-�?i9�L��?�Unknown
wHostMaxPool""sequential/max_pooling1d_1/MaxPool(133333^@933333^@A33333^@I33333^@atK� ��?igP��$�?�Unknown
pHostBiasAdd"sequential/conv1d_1/BiasAdd(1fffff�]@9fffff�]@Afffff�]@Ifffff�]@a��XR�D�?iU����?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1     `\@9     `\@A     `\@I     `\@aE129;�?i}��	�?�Unknown
�HostReluGrad"*gradient_tape/sequential/conv1d_1/ReluGrad(1�����LW@9�����LW@A�����LW@I�����LW@a��*�?ij���(e�?�Unknown
^HostGatherV2"GatherV2(1333333W@9333333W@A333333W@I333333W@a�΂Ά?i��0�`��?�Unknown
�HostBiasAddGrad"5gradient_tape/sequential/conv1d_1/BiasAdd/BiasAddGrad(1fffff�R@9fffff�R@Afffff�R@Ifffff�R@a�Fa��5�?i�#v7	�?�Unknown
~HostReluGrad"(gradient_tape/sequential/conv1d/ReluGrad(1�����YI@9�����YI@A�����YI@I�����YI@a�`���x?i� �(;�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(133333sH@933333sH@A33333sH@I33333sH@aqe@�x?id�NCk�?�Unknown
jHostRelu"sequential/conv1d_1/Relu(1fffff�E@9fffff�E@Afffff�E@Ifffff�E@a���Gu?ik1���?�Unknown
hHostRelu"sequential/conv1d/Relu(1������=@9������=@A������=@I������=@aެ9��Jm?i�T9���?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(133333�<@933333�<@A33333�<@I33333�<@a�2��6l?i��/��?�Unknown
oHostSoftmax"sequential/dense_1/Softmax(1      4@9      4@A      4@I      4@a�$*èc?i"�����?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1������3@9������3@A      0@I      0@a��vckt_?isՌ����?�Unknown
dHostDataset"Iterator::Model(1fffff�4@9fffff�4@A333333/@I333333/@a"w`��^?i������?�Unknown
`HostGatherV2"
GatherV2_1(1������,@9������,@A������,@I������,@a5m��\?i�J���?�Unknown
i HostWriteSummary"WriteSummary(1������&@9������&@A������&@I������&@a�.rq7V?i�J|�?�Unknown�
{!HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1������%@9������%@A������%@I������%@ay\��mU?i���V�%�?�Unknown
}"HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      $@9      $@A      $@I      $@a�$*èS?i���/�?�Unknown
t#Host_FusedMatMul"sequential/dense_1/BiasAdd(1������#@9������#@A������#@I������#@a`�$?ovS?i^�,�W9�?�Unknown
�$HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1������#@9������#@A������#@I������#@a�`DS?i����B�?�Unknown
q%HostCast"sequential/dropout/dropout/Cast(1      !@9      !@A      !@I      !@a��ٵP?iic�TK�?�Unknown
�&HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1������ @9������ @A������ @I������ @a:{�-��P?i'���S�?�Unknown
�'HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1       @9       @A       @I       @a��vcktO?i����s[�?�Unknown
Z(HostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a�LJk��M?ib��:�b�?�Unknown
�)HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      @9      @A      @I      @az7?�$}M?i0�Kj�?�Unknown
�*HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1ffffff@9ffffff@Affffff@Iffffff@a�����K?i�H�%Fq�?�Unknown
�+HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1������@9������@A������@I������@a����WJ?i��5�w�?�Unknown
g,HostStridedSlice"strided_slice(1      @9      @A      @I      @aGc�@��I?i���?~�?�Unknown
�-HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1ffffff"@9ffffff"@A������@I������@aNł�)I?i�g�@���?�Unknown
�.HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      @9      @A      @I      @a.���P�G?i7	p��?�Unknown
�/HostResourceApplyAdam"$Adam/Adam/update_7/ResourceApplyAdam(1������@9������@A������@I������@a��̨2G?i�1<�<��?�Unknown
�0HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a�΂�F?id�?��?�Unknown
s1HostDataset"Iterator::Model::ParallelMapV2(1333333@9333333@A333333@I333333@a�dKX��D?i=��%��?�Unknown
�2HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1      @9      @A      @I      @a�$*èC?i�o���?�Unknown
�3HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1333333@9333333@A333333@I333333@ar���,�@?i�f*J��?�Unknown
V4HostSum"Sum_2(1������@9������@A������@I������@a:{�-��@?iI�ck��?�Unknown
x5HostDataset"#Iterator::Model::ParallelMapV2::Zip(1�����LA@9�����LA@Affffff@Iffffff@af�o�@?i�̿�r��?�Unknown
�6HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?a��s-O<?i�0n����?�Unknown
�7HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a��s-O<?ia�����?�Unknown
v8HostCast"$sparse_categorical_crossentropy/Cast(1333333@9333333@A333333@I333333@a��z��:?i���޶�?�Unknown
l9HostIteratorGetNext"IteratorGetNext(1ffffff
@9ffffff
@Affffff
@Iffffff
@ax��>�9?i�ˇ��?�Unknown
�:HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1ffffff
@9ffffff
@Affffff
@Iffffff
@ax��>�9?is���Z��?�Unknown
�;HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1������	@9������	@A������	@I������	@aNł�)9?i�-���?�Unknown
e<Host
LogicalAnd"
LogicalAnd(1333333@9333333@A333333@I333333@a�΂�6?iwҽ�Y��?�Unknown�
�=HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@aM�l��6?i ����?�Unknown
Y>HostPow"Adam/Pow(1������@9������@A������@I������@a�yVb;5?i��2����?�Unknown
�?HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1������@9������@A������@I������@a�yVb;5?i��u\i��?�Unknown
o@HostMul"sequential/dropout/dropout/Mul(1      @9      @A      @I      @a�$*è3?i�z�t���?�Unknown
[AHostAddV2"Adam/add(1������@9������@A������@I������@a����L1?i�n��?�Unknown
tBHostAssignAddVariableOp"AssignAddVariableOp(1������@9������@A������@I������@a����L1?i���1��?�Unknown
bCHostDivNoNan"div_no_nan_1(1������@9������@A������@I������@a����L1?i�1�D[��?�Unknown
vDHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1������ @9������ @A������ @I������ @a:{�-��0?i��>�k��?�Unknown
[EHostPow"
Adam/Pow_1(1������ @9������ @A������ @I������ @a:{�-��0?i,��%|��?�Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_2(1������ @9������ @A������ @I������ @a:{�-��0?i[`�����?�Unknown
XGHostEqual"Equal(1       @9       @A       @I       @a��vckt/?iŗ@݃��?�Unknown
XHHostCast"Cast_3(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�LJk��-?ijL�a��?�Unknown
~IHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1�������?9�������?A�������?I�������?a��s-O,?iI~��&��?�Unknown
�JHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������%@9������%@A333333�?I333333�?a��z��*?ic-Ƶ���?�Unknown
TKHostMul"Mul(1333333�?9333333�?A333333�?I333333�?a��z��*?i}ܭ~~��?�Unknown
wLHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a��z��*?i���G*��?�Unknown
}MHostMul",gradient_tape/sequential/dropout/dropout/Mul(1333333�?9333333�?A333333�?I333333�?a��z��*?i�:}���?�Unknown
�NHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?aNł�))?igu�h��?�Unknown
OHostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1      �?9      �?A      �?I      �?a.���P�'?i�~$���?�Unknown
�PHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1      �?9      �?A      �?I      �?a.���P�'?i&���[��?�Unknown
]QHostCast"Adam/Cast_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aM�l��&?i������?�Unknown
�RHostReadVariableOp"4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aM�l��&?i��/��?�Unknown
VSHostCast"Cast(1�������?9�������?A�������?I�������?amO@�r$?i���Pc��?�Unknown
uTHostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?amO@�r$?i�Or���?�Unknown
�UHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?amO@�r$?i��5����?�Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_4(1333333�?9333333�?A333333�?I333333�?a���s�"?i	p���?�Unknown
qWHostMul" sequential/dropout/dropout/Mul_1(1333333�?9333333�?A333333�?I333333�?a���s�"?iI6��M��?�Unknown
�XHostReadVariableOp"6sequential/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp(1�������?9�������?A�������?I�������?a����L!?i���Nb��?�Unknown
XYHostCast"Cast_2(1�������?9�������?A�������?I�������?a��s-O?i�m`�D��?�Unknown
tZHostReadVariableOp"Adam/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?aNł�)?i݃���?�Unknown
v[HostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?aNł�)?i�Xg���?�Unknown
v\HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aM�l��?il-匇��?�Unknown
X]HostCast"Cast_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?aM�l��?i��q�7��?�Unknown
�^HostReadVariableOp"(sequential/conv1d/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aM�l��?i6T�����?�Unknown
�_HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aM�l��?i������?�Unknown
�`HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1ffffff�?9ffffff�?Affffff�?Iffffff�?aM�l��?i {#H��?�Unknown
oaHostReadVariableOp"Adam/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a���s�?i������?�Unknown
`bHostDivNoNan"
div_no_nan(1333333�?9333333�?A333333�?I333333�?a���s�?i@�Qv��?�Unknown
ycHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a���s�?i����?�Unknown
�dHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a���s�?i������?�Unknown
veHostAssignAddVariableOp"AssignAddVariableOp_1(1      �?9      �?A      �?I      �?a��vckt?i[K9�!��?�Unknown
wfHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      �?9      �?A      �?I      �?a��vckt?i6�洟��?�Unknown
�gHostReadVariableOp"*sequential/conv1d_1/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?a��vckt?ig����?�Unknown
�hHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a��vckt?i��AX���?�Unknown
aiHostIdentity"Identity(1�������?9�������?A�������?I�������?aNł�)	?i     �?�Unknown�