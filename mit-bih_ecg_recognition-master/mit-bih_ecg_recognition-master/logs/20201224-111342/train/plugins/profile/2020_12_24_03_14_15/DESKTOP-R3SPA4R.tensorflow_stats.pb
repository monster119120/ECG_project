"�Y
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
BHostIDLE"IDLE1����F�@A����F�@a�*i��?i�*i��?�Unknown
�HostConv2DBackpropFilter";gradient_tape/sequential/conv1d/conv1d/Conv2DBackpropFilter(1����Y��@9����Y��@A����Y��@I����Y��@aKH2a�N�?i�k�7���?�Unknown
�HostBiasAddGrad"3gradient_tape/sequential/conv1d/BiasAdd/BiasAddGrad(1     ��@9     ��@A     ��@I     ��@a.���8̴?i�E(V|��?�Unknown
lHostConv2D"sequential/conv1d/conv1d(133333Q�@933333Q�@A33333Q�@I33333Q�@aM���巫?i3E��E�?�Unknown
^HostGatherV2"GatherV2(133333y�@933333y�@A33333y�@I33333y�@a'3�&R�?i3����?�Unknown
uHostMaxPool" sequential/max_pooling1d/MaxPool(1�������@9�������@A�������@I�������@a�����?i4�ǅ��?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1����̰�@9����̰�@A����̰�@I����̰�@a�ע�F�?i��J"�?�Unknown�
�	HostMaxPoolGrad":gradient_tape/sequential/max_pooling1d/MaxPool/MaxPoolGrad(1fffff�n@9fffff�n@Afffff�n@Ifffff�n@a��"�?i1��8f�?�Unknown
�
HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     �j@9     �j@A     �j@I     �j@a�'�s&}?i`ӗd��?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1     d@9     d@A     d@I     d@af֐Z�u?i�LZR��?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1������c@9������c@A������c@I������c@a 9��u?i�B���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1�����ia@9�����ia@A�����ia@I�����ia@ap!i_s?i��AI7�?�Unknown
nHostBiasAdd"sequential/conv1d/BiasAdd(1�����L]@9�����L]@A�����L]@I�����L]@a� �
p?iÝk|K>�?�Unknown
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1����̬X@9����̬X@A����̬X@I����̬X@a����k?i�.;dOY�?�Unknown
hHostRelu"sequential/conv1d/Relu(133333�J@933333�J@A33333�J@I33333�J@aA��̊�]?iC��)h�?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1333333G@9333333G@A333333G@I333333G@a�	�"~fY?iH�h�t�?�Unknown
~HostReluGrad"(gradient_tape/sequential/conv1d/ReluGrad(1fffff�D@9fffff�D@Afffff�D@Ifffff�D@a]&�qǛV?i��kL��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1�����LB@9�����LB@A�����LB@I�����LB@a~��	T?i�i����?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1�����L>@9�����L>@A�����L>@I�����L>@aRg=�P?i���`��?�Unknown
oHostSoftmax"sequential/dense_1/Softmax(1     �;@9     �;@A     �;@I     �;@a�B]4�N?i,*7���?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������:@9������:@A������:@I������:@a����WM?i�:�=��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(133333�2@933333�2@A33333�2@I33333�2@aG-5�;yD?iY�)\��?�Unknown
dHostDataset"Iterator::Model(1ffffff7@9ffffff7@Afffff�1@Ifffff�1@a���l�C?i�VB��?�Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1������1@9������1@A������1@I������1@a�yB�|C?iu���!��?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(133333�1@933333�1@A33333�1@I33333�1@aP�d�`C?i�ś����?�Unknown
�HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1������1@9������1@A������1@I������1@a�*P��DC?i�Y�˹�?�Unknown
qHostCast"sequential/dropout/dropout/Cast(13333330@93333330@A3333330@I3333330@aݲ,���A?i��/:��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1������3@9������3@A������/@I������/@al��kLA?iG��J���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1������+@9������+@A������+@I������+@aIr^�7>?i��)BT��?�Unknown
iHostWriteSummary"WriteSummary(1ffffff'@9ffffff'@Affffff'@Iffffff'@a�w��9?i������?�Unknown�
` HostGatherV2"
GatherV2_1(1������$@9������$@A������$@I������$@a����Í6?i�"4�Y��?�Unknown
�!HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1������!@9������!@A������!@I������!@a�*P��D3?i���i���?�Unknown
�"HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1333333 @9333333 @A333333 @I333333 @aݲ,���1?i3�����?�Unknown
�#HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1ffffff@9ffffff@Affffff@Iffffff@a�\�@�0?iÝ���?�Unknown
e$Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@aۡ��/?iM�0 ��?�Unknown�
g%HostStridedSlice"strided_slice(1ffffff@9ffffff@Affffff@Iffffff@aۡ��/?i׀[����?�Unknown
Z&HostArgMax"ArgMax(1������@9������@A������@I������@aIr^�7.?i�g�����?�Unknown
['HostAddV2"Adam/add(1333333@9333333@A333333@I333333@a�	�"~f)?iٕ�dk��?�Unknown
v(HostCast"$sparse_categorical_crossentropy/Cast(1333333@9333333@A333333@I333333@a�	�"~f)?i�õ���?�Unknown
�)HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1ffffff@9ffffff@Affffff@Iffffff@a|7�D�(?i2�1���?�Unknown
s*HostDataset"Iterator::Model::ParallelMapV2(1      @9      @A      @I      @aE5�)((?iu�����?�Unknown
�+HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1333333@9333333@A333333@I333333@a��=��5'?iOs�~��?�Unknown
V,HostSum"Sum_2(1������@9������@A������@I������@a�`�0��&?i�'�O���?�Unknown
�-HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������@9������@A������@I������@a�`�0��&?i�6��W��?�Unknown
x.HostDataset"#Iterator::Model::ParallelMapV2::Zip(1�����B@9�����B@Affffff@Iffffff@a���U&?i����?�Unknown
�/HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1ffffff#@9ffffff#@Affffff@Iffffff@a���U&?iIjc"��?�Unknown
�0HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a��I�B�$?i���k��?�Unknown
�1HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a0��>&%$?i���	���?�Unknown
�2HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@aAVV���!?i�lS���?�Unknown
�3HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1ffffff@9ffffff@Affffff@Iffffff@aAVV���!?iUXۜ���?�Unknown
]4HostCast"Adam/Cast_1(1      @9      @A      @I      @axMz�!?i�(����?�Unknown
o5HostMul"sequential/dropout/dropout/Mul(1333333@9333333@A333333@I333333@a�ȯ�]!?i�sZ*��?�Unknown
~6HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a�\�@� ?iK9jn ��?�Unknown
�7HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff�?Affffff@Iffffff�?a�\�@� ?i�y�*��?�Unknown
Y8HostPow"Adam/Pow(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�?xec�?i�*����?�Unknown
l9HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a�$+��&?i.L%���?�Unknown
�:HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1      @9      @A      @I      @a4��s�F?iS��Z���?�Unknown
t;HostAssignAddVariableOp"AssignAddVariableOp(1������@9������@A������@I������@a|�?i��2�z��?�Unknown
�<HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aV�C���?i����)��?�Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_2(1333333@9333333@A333333@I333333@a�E��_?i�s�����?�Unknown
[>HostPow"
Adam/Pow_1(1ffffff@9ffffff@Affffff@Iffffff@a0��>&%?i�k(s��?�Unknown
t?HostReadVariableOp"Adam/Cast/ReadVariableOp(1������ @9������ @A������ @I������ @a����d?i�X�1��?�Unknown
X@HostEqual"Equal(1������ @9������ @A������ @I������ @a����d?i3FbW���?�Unknown
qAHostMul" sequential/dropout/dropout/Mul_1(1������ @9������ @A������ @I������ @a����d?i�3�|,��?�Unknown
oBHostReadVariableOp"Adam/ReadVariableOp(1       @9       @A       @I       @axMz�?i��Ѡ���?�Unknown
�CHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1       @9       @A       @I       @axMz�?i���D��?�Unknown
�DHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�\�@�?i������?�Unknown
�EHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      '@9      '@A�������?I�������?a��kW�?iDD�H��?�Unknown
`FHostDivNoNan"
div_no_nan(1�������?9�������?A�������?I�������?a��kW�?i��&'���?�Unknown
GHostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1�������?9�������?A�������?I�������?a��kW�?i��cGD��?�Unknown
VHHostCast"Cast(1333333�?9333333�?A333333�?I333333�?a����?i��e���?�Unknown
vIHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1�������?9�������?A�������?I�������?aZ��*?if+�+��?�Unknown
XJHostCast"Cast_2(1      �?9      �?A      �?I      �?a4��s�F
?ix�\����?�Unknown
}KHostMul",gradient_tape/sequential/dropout/dropout/Mul(1      �?9      �?A      �?I      �?a4��s�F
?i��:����?�Unknown
vLHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a|7�D�?ihN�_��?�Unknown
�MHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a|7�D�?iFYa����?�Unknown
vNHostAssignAddVariableOp"AssignAddVariableOp_4(1�������?9�������?A�������?I�������?a�`�0��?i����?�Unknown
vOHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?a�E��_?ie[(q��?�Unknown
TPHostMul"Mul(1333333�?9333333�?A333333�?I333333�?a�E��_?iڙ�,���?�Unknown
uQHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�E��_?iO�$B��?�Unknown
wRHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�E��_?i��Wm��?�Unknown
�SHostReadVariableOp"4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�*P��D?i�Vk���?�Unknown
aTHostIdentity"Identity(1      �?9      �?A      �?I      �?axMz�?i@} ��?�Unknown�
XUHostCast"Cast_3(1�������?9�������?A�������?I�������?a��kW��>i�^�?��?�Unknown
bVHostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a��kW��>i�a}�~��?�Unknown
�WHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1�������?9�������?A�������?I�������?a��kW��>i������?�Unknown
�XHostReadVariableOp"(sequential/conv1d/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?aZ��*�>i<:����?�Unknown
vYHostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff�?9ffffff�?Affffff�?Iffffff�?a|7�D��>i��y�&��?�Unknown
XZHostCast"Cast_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?a|7�D��>i��W��?�Unknown
�[HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a|7�D��>i�'����?�Unknown
�\HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a|7�D��>i�����?�Unknown
w]HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      �?9      �?A      �?I      �?axMz��>i�e����?�Unknown
y^HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?axMz��>i     �?�Unknown*�X
�HostConv2DBackpropFilter";gradient_tape/sequential/conv1d/conv1d/Conv2DBackpropFilter(1����Y��@9����Y��@A����Y��@I����Y��@a� 7F&��?i� 7F&��?�Unknown
�HostBiasAddGrad"3gradient_tape/sequential/conv1d/BiasAdd/BiasAddGrad(1     ��@9     ��@A     ��@I     ��@a�-.����?i	���H��?�Unknown
lHostConv2D"sequential/conv1d/conv1d(133333Q�@933333Q�@A33333Q�@I33333Q�@a���o��?i�!7ζ��?�Unknown
^HostGatherV2"GatherV2(133333y�@933333y�@A33333y�@I33333y�@a�9��:�?i�h��2�?�Unknown
uHostMaxPool" sequential/max_pooling1d/MaxPool(1�������@9�������@A�������@I�������@a�ǥ/�?i��?��?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1����̰�@9����̰�@A����̰�@I����̰�@a��0d
@�?i�F*���?�Unknown�
�HostMaxPoolGrad":gradient_tape/sequential/max_pooling1d/MaxPool/MaxPoolGrad(1fffff�n@9fffff�n@Afffff�n@Ifffff�n@a���^��?i���C�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     �j@9     �j@A     �j@I     �j@a��$��?i��~���?�Unknown
o	Host_FusedMatMul"sequential/dense/Relu(1     d@9     d@A     d@I     d@a��i�Ȃ?i��	����?�Unknown
y
HostMatMul"%gradient_tape/sequential/dense/MatMul(1������c@9������c@A������c@I������c@a�2�u#��?iy���<�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1�����ia@9�����ia@A�����ia@I�����ia@a٩��M�?i ���}�?�Unknown
nHostBiasAdd"sequential/conv1d/BiasAdd(1�����L]@9�����L]@A�����L]@I�����L]@ai�:�m{?i5?y���?�Unknown
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1����̬X@9����̬X@A����̬X@I����̬X@aʭn��w?i�����?�Unknown
hHostRelu"sequential/conv1d/Relu(133333�J@933333�J@A33333�J@I33333�J@a�<���:i?i��G+��?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1333333G@9333333G@A333333G@I333333G@a|�#	�e?i���P��?�Unknown
~HostReluGrad"(gradient_tape/sequential/conv1d/ReluGrad(1fffff�D@9fffff�D@Afffff�D@Ifffff�D@aG��<�Tc?i��:8%�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1�����LB@9�����LB@A�����LB@I�����LB@a�v�·!a?ix��Y6�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1�����L>@9�����L>@A�����L>@I�����L>@a�tt-�]\?iP2[��D�?�Unknown
oHostSoftmax"sequential/dense_1/Softmax(1     �;@9     �;@A     �;@I     �;@a�8?��Y?i���hQ�?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������:@9������:@A������:@I������:@a!����Y?i��\l�]�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(133333�2@933333�2@A33333�2@I33333�2@a.n����Q?i@t�6�f�?�Unknown
dHostDataset"Iterator::Model(1ffffff7@9ffffff7@Afffff�1@Ifffff�1@a^����P?i a$o�?�Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1������1@9������1@A������1@I������1@a�����P?iA1�jw�?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(133333�1@933333�1@A33333�1@I33333�1@a��g}�P?i�7��?�Unknown
�HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1������1@9������1@A������1@I������1@aх.G�yP?iF|����?�Unknown
qHostCast"sequential/dropout/dropout/Cast(13333330@93333330@A3333330@I3333330@a�G!��TN?i��?���?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1������3@9������3@A������/@I������/@a�XY�&�M?i�����?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1������+@9������+@A������+@I������+@a�qu��I?iZw\*`��?�Unknown
iHostWriteSummary"WriteSummary(1ffffff'@9ffffff'@Affffff'@Iffffff'@a0	���E?i\}@(ڢ�?�Unknown�
`HostGatherV2"
GatherV2_1(1������$@9������$@A������$@I������$@a[Eܡ�HC?im��c���?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1������!@9������!@A������!@I������!@aх.G�y@?i���ʫ�?�Unknown
� HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1333333 @9333333 @A333333 @I333333 @a�G!��T>?i7�}���?�Unknown
�!HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1ffffff@9ffffff@Affffff@Iffffff@a�r�c�u<?i�9/$��?�Unknown
e"Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a��9'@�:?i! �v��?�Unknown�
g#HostStridedSlice"strided_slice(1ffffff@9ffffff@Affffff@Iffffff@a��9'@�:?iM%�ɹ�?�Unknown
Z$HostArgMax"ArgMax(1������@9������@A������@I������@a�qu��9?i�����?�Unknown
[%HostAddV2"Adam/add(1333333@9333333@A333333@I333333@a|�#	�5?iE+����?�Unknown
v&HostCast"$sparse_categorical_crossentropy/Cast(1333333@9333333@A333333@I333333@a|�#	�5?i�<�r��?�Unknown
�'HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1ffffff@9ffffff@Affffff@Iffffff@a��qO�4?i��*���?�Unknown
s(HostDataset"Iterator::Model::ParallelMapV2(1      @9      @A      @I      @aE'��r�4?i�}����?�Unknown
�)HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1333333@9333333@A333333@I333333@av82��3?iW����?�Unknown
V*HostSum"Sum_2(1������@9������@A������@I������@aAN�x3?i�ݎ��?�Unknown
�+HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������@9������@A������@I������@aAN�x3?i�t�����?�Unknown
x,HostDataset"#Iterator::Model::ParallelMapV2::Zip(1�����B@9�����B@Affffff@Iffffff@a�Ij5�3?i0"�a��?�Unknown
�-HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1ffffff#@9ffffff#@Affffff@Iffffff@a�Ij5�3?iy�k8���?�Unknown
�.HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@al�ы�1?i�
�i���?�Unknown
�/HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a�t���91?i�)ş��?�Unknown
�0HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a3?y��.?i깜�	��?�Unknown
�1HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1ffffff@9ffffff@Affffff@Iffffff@a3?y��.?i>Jt7���?�Unknown
]2HostCast"Adam/Cast_1(1      @9      @A      @I      @adP=��-?i������?�Unknown
o3HostMul"sequential/dropout/dropout/Mul(1333333@9333333@A333333@I333333@a�auJ5-?iiRܧ��?�Unknown
~4HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a�r�c�u,?i@PX5o��?�Unknown
�5HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff�?Affffff@Iffffff�?a�r�c�u,?i�^�6��?�Unknown
Y6HostPow"Adam/Pow(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�����(?it7]����?�Unknown
l7HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a�5�|7'?iӪ%u5��?�Unknown
�8HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1      @9      @A      @I      @aK�m��w&?i�S���?�Unknown
t9HostAssignAddVariableOp"AssignAddVariableOp(1������@9������@A������@I������@a�/��8$?i�z���?�Unknown
�:HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?R�\"�"?i{����?�Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_2(1333333@9333333@A333333@I333333@aoc��h�!?iaw_�+��?�Unknown
[<HostPow"
Adam/Pow_1(1ffffff@9ffffff@Affffff@Iffffff@a�t���9!?i�O>?��?�Unknown
t=HostReadVariableOp"Adam/Cast/ReadVariableOp(1������ @9������ @A������ @I������ @a.�*wt?i1]�:��?�Unknown
X>HostEqual"Equal(1������ @9������ @A������ @I������ @a.�*wt?i����6��?�Unknown
q?HostMul" sequential/dropout/dropout/Mul_1(1������ @9������ @A������ @I������ @a.�*wt?i
{)2��?�Unknown
o@HostReadVariableOp"Adam/ReadVariableOp(1       @9       @A       @I       @adP=��?i�C��!��?�Unknown
�AHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1       @9       @A       @I       @adP=��?i�}�y��?�Unknown
�BHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�r�c�u?iE�:&���?�Unknown
�CHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      '@9      '@A�������?I�������?a'� �?i2�"����?�Unknown
`DHostDivNoNan"
div_no_nan(1�������?9�������?A�������?I�������?a'� �?i�
����?�Unknown
EHostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1�������?9�������?A�������?I�������?a'� �?i��8|��?�Unknown
VFHostCast"Cast(1333333�?9333333�?A333333�?I333333�?a�����v?iz�?�G��?�Unknown
vGHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1�������?9�������?A�������?I�������?a���86�?iiJ���?�Unknown
XHHostCast"Cast_2(1      �?9      �?A      �?I      �?aK�m��w?i��f���?�Unknown
}IHostMul",gradient_tape/sequential/dropout/dropout/Mul(1      �?9      �?A      �?I      �?aK�m��w?iI�$o��?�Unknown
vJHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��qO�?i:0����?�Unknown
�KHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��qO�?i+�����?�Unknown
vLHostAssignAddVariableOp"AssignAddVariableOp_4(1�������?9�������?A�������?I�������?aAN�x?i�1�oZ��?�Unknown
vMHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?aoc��h�?i��;;���?�Unknown
TNHostMul"Mul(1333333�?9333333�?A333333�?I333333�?aoc��h�?i�݀z��?�Unknown
uOHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aoc��h�?iv3��	��?�Unknown
wPHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aoc��h�?ii�����?�Unknown
�QHostReadVariableOp"4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp(1�������?9�������?A�������?I�������?aх.G�y?i�µl��?�Unknown
aRHostIdentity"Identity(1      �?9      �?A      �?I      �?adP=��?i���@���?�Unknown�
XSHostCast"Cast_3(1�������?9�������?A�������?I�������?a'� �
?iH�8��?�Unknown
bTHostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a'� �
?i���l��?�Unknown
�UHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1�������?9�������?A�������?I�������?a'� �
?i4� ����?�Unknown
�VHostReadVariableOp"(sequential/conv1d/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a���86�?i+���8��?�Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��qO�?i��7����?�Unknown
XXHostCast"Cast_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��qO�?iTui���?�Unknown
�YHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��qO�?i��J4��?�Unknown
�ZHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��qO�?i��+���?�Unknown
w[HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      �?9      �?A      �?I      �?adP=���>i�q����?�Unknown
y\HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?adP=���>i      �?�Unknown