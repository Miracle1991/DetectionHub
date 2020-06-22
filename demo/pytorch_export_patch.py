"""
Includes updates required for PyTorch to export Faster RCNN model to ONNX.
These are submitted as PRs, but have not been merged yet.
"""
import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _parse_arg, _maybe_get_const, _unimplemented, _is_value, cast_pytorch_to_onnx, _unpack_list
from torch.onnx.symbolic_opset9 import _cast_Float, _cast_Long, _cast_Int, t, index_select, squeeze
import torch.onnx.symbolic_opset10
import warnings

@parse_args('v', 'v', 'i', 'i', 'i')
def topk(g, self, k, dim, largest, sorted, out=None):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported for topk")
    if not largest:
        _unimplemented("TopK", "Ascending TopK is not supported")
    k_value = _maybe_get_const(k, 'i')
    if not _is_value(k_value):
        k = g.op("Constant", value_t=torch.tensor(k_value, dtype=torch.int64))
    from torch.onnx.symbolic_opset9 import unsqueeze
    k = unsqueeze(g, k_value, 0)
    return g.op("TopK", self, k, axis_i=dim, outputs=2)


def slice_opset10(g, self, dim, start, end, step):
    if start.node().kind() != 'onnx::Constant' or \
            end.node().kind() != 'onnx::Constant' or dim.node().kind() != 'onnx::Constant' or \
            step.node().kind() != 'onnx::Constant':
        start_unsqueezed = g.op("Unsqueeze", start, axes_i=[0])
        end_unsqueezed = g.op("Unsqueeze", end, axes_i=[0])
        dim_unsqueezed = g.op("Unsqueeze", dim, axes_i=[0])
        step_unsqueezed = g.op("Unsqueeze", step, axes_i=[0])
        return g.op("Slice", self, start_unsqueezed, end_unsqueezed, dim_unsqueezed, step_unsqueezed)
    else:
        start = _parse_arg(start, 'i')
        end = _parse_arg(end, 'i')
        dim = _parse_arg(dim, 'i')
        step = _parse_arg(step, 'i')
        start_tensor = g.op('Constant', value_t=torch.tensor([start], dtype=torch.long))
        end_tensor = g.op('Constant', value_t=torch.tensor([end], dtype=torch.long))
        dim_tensor = g.op('Constant', value_t=torch.tensor([dim], dtype=torch.long))
        step_tensor = g.op('Constant', value_t=torch.tensor([step], dtype=torch.long))
        return g.op("Slice", self, start_tensor, end_tensor, dim_tensor, step_tensor)


def upsample_nearest2d(g, input, output_size):
    output_size = _maybe_get_const(output_size, 'is')

    if _is_value(output_size):
        div_lhs = g.op('Cast', output_size, to_i=cast_pytorch_to_onnx['Float'])
        div_rhs = g.op('Cast',
            g.op('Slice',
                g.op('Shape', input),
                g.op('Constant', value_t=torch.tensor([2], dtype=torch.long)),
                g.op('Constant', value_t=torch.tensor([4], dtype=torch.long))),
            to_i=cast_pytorch_to_onnx['Float'])

        scales = g.op('Concat', g.op('Constant', value_t=torch.tensor([1., 1.])), g.op('Div', div_lhs, div_rhs), axis_i=0)
    else:
        height_scale = float(output_size[-2]) / input.type().sizes()[-2]
        width_scale = float(output_size[-1]) / input.type().sizes()[-1]
        scales = g.op("Constant", value_t=torch.tensor([1., 1., height_scale,
                                                        width_scale]))

    return g.op("Resize", input, scales, #'Upsample' for opset 9
                mode_s="nearest")


def min_opset9(g, self, dim_or_y=None, keepdim=None):
    # torch.min(input)
    if dim_or_y is None and keepdim is None:
        return _cast_Long(g, g.op("ReduceMin", _cast_Int(g, self, False), keepdims_i=0), False)
    # torch.min(input, other)
    if keepdim is None:
        return g.op("Min", self, dim_or_y)
    # torch.min(input, dim, keepdim)
    else:
        dim = sym_help._get_const(dim_or_y, 'i', 'dim')
        keepdim = sym_help._get_const(keepdim, 'i', 'keepdim')
        min = g.op("ReduceMin", self, axes_i=[dim], keepdims_i=keepdim)
        indices = g.op('ArgMin', self, axis_i=dim, keepdims_i=keepdim)
        return min, indices


def nonzero(g, input):
    return t(g, g.op('NonZero', _cast_Float(g, input, False)))


def _is_packed_list(list_value):
    list_node = list_value.node()
    return list_node.kind() == "prim::ListConstruct"


def index(g, self, index):
    if _is_packed_list(index):
        indices = sym_help._unpack_list(index)
    else:
        indices = [index]

    if len(indices) == 1:
        if indices[0].type().scalarType() == "Byte":
            indices[0] = squeeze(g, nonzero(g, indices[0]), dim=1)
        return index_select(g, self, 0, indices[0])
    else:
        raise NotImplementedError("Unsupported aten::index operator with more than 1 indices tensor. ")


from torch.onnx.utils import OperatorExportTypes
def _run_symbolic_function(g, n, inputs, env, operator_export_type=OperatorExportTypes.ONNX):
    # NB: Returning None means the node gets cloned as is into
    # the new graph
    try:
        from torch.onnx.symbolic_helper import _export_onnx_opset_version as opset_version
        import torch.onnx.symbolic_registry as sym_registry

        sym_registry.register_version('', opset_version)

        # See Note [Export inplace]
        # TODO: I think this is not necessary anymore
        if n.kind().endswith('_'):
            ns_op_name = n.kind()[:-1]
        else:
            ns_op_name = n.kind()
        ns, op_name = ns_op_name.split("::")

        if ns == "onnx":
            # Use the original node directly
            return None

        elif ns == "aten":
            is_exportable_aten_op = sym_registry.is_registered_op(op_name, '', opset_version)
            is_onnx_aten_export = operator_export_type == OperatorExportTypes.ONNX_ATEN
            is_aten_fallback_export = operator_export_type == OperatorExportTypes.ONNX_ATEN_FALLBACK
            if is_onnx_aten_export or (not is_exportable_aten_op and is_aten_fallback_export):
                # Direct ATen export requested
                attrs = {k + "_" + n.kindOf(k)[0]: n[k] for k in n.attributeNames()}
                outputs = n.outputsSize()
                attrs["outputs"] = outputs
                return _graph_at(g, op_name, *inputs, aten=True, **attrs)

            else:
                # Export it regularly
                attrs = {k: n[k] for k in n.attributeNames()}
                if not is_exportable_aten_op:
                    warnings.warn("ONNX export failed on ATen operator {} because "
                                  "torch.onnx.symbolic_opset{}.{} does not exist"
                                  .format(op_name, opset_version, op_name))
                op_fn = sym_registry.get_registered_op(op_name, '', opset_version)
                return op_fn(g, *inputs, **attrs)

        elif ns == "prim":
            if op_name == "Constant" and not n.mustBeNone():
                if n.kindOf("value") == "t":
                    return g.op("Constant", value_t=n["value"])
                elif n.kindOf("value") == "is":
                    value = torch.stack([torch.tensor(v) for v in n["value"]]) if n["value"] else []
                    return g.op("Constant", value_t=value)
                elif n.output().type().kind() == "DeviceObjType":
                    return None
                else:
                    raise RuntimeError("Unsupported prim::Constant kind: `{}`. Send a bug report.".format(
                        n.kindOf("value")))
            elif n.mustBeNone() or op_name == "ListConstruct" or op_name == "ListUnpack":
                # None is not an ONNX operator; keep it as None
                # let the exporter handle finally eliminating these

                # For ListConstruct/ListUnpack, it will be erased in the ONNX peephole pass
                return None
            elif op_name == 'Loop' or op_name == 'If':
                new_op_outputs = g.op(op_name, *inputs, outputs=n.outputsSize())
                new_node = new_op_outputs[0].node() if n.outputsSize() > 1 else new_op_outputs.node()
                for b in n.blocks():
                    new_block = new_node.addBlock()
                    torch._C._jit_pass_onnx_block(b, new_block, operator_export_type, env)
                return new_op_outputs
            else:
                # TODO: we sould lift prim's symbolic out
                symbolic_name = 'prim_' + op_name
                is_exportable = sym_registry.is_registered_op(symbolic_name, '', opset_version)
                if not is_exportable:
                    warnings.warn("ONNX export failed on primitive operator {}; please report a bug".format(op_name))
                symbolic_fn = sym_registry.get_registered_op(symbolic_name, '', opset_version)
                attrs = {k: n[k] for k in n.attributeNames()}
                return symbolic_fn(g, *inputs, **attrs)

        # custom ops
        elif sym_registry.is_registered_version(ns, opset_version):
            if not sym_registry.is_registered_op(op_name, ns, opset_version):
                warnings.warn("ONNX export failed on custom operator {}::{} because "
                              "torch.onnx.symbolic_opset{}.{} does not exist. "
                              "Have you registered your symbolic function with "
                              "torch.onnx.register_custom_op_symbolic(symbolic_name, symbolic_fn)?"
                              .format(ns, op_name, opset_version, op_name))
            symbolic_fn = sym_registry.get_registered_op(op_name, ns, opset_version)
            attrs = {k: n[k] for k in n.attributeNames()}
            return symbolic_fn(g, *inputs, **attrs)

        else:
            warnings.warn("ONNX export failed on an operator with unrecognized namespace {}::{}; "
                          "If you are trying to export a custom operator, make sure you registered "
                          "it with the right domain and version."
                          "Otherwise please report a bug".format(ns, op_name))
            return None

    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch.
        # Otherwise, the backtrace will have the clues you need.
        e.args = ("{} (occurred when translating {})".format(e.args[0], op_name), )
        raise

torch.onnx.utils._run_symbolic_function = _run_symbolic_function

torch.onnx.symbolic_opset9.min = min_opset9
torch.onnx.symbolic_opset9.nonzero = nonzero
torch.onnx.symbolic_opset9.index = index
torch.onnx.symbolic_opset10.topk = topk
torch.onnx.symbolic_opset10.slice = slice_opset10
torch.onnx.symbolic_opset10.upsample_nearest2d = upsample_nearest2d


def register_custom_op():
    # experimenting custom op registration.
    from torch.onnx.symbolic_helper import parse_args
    from torch.onnx.symbolic_opset9 import select, unsqueeze, squeeze, _cast_Long
    @parse_args('v', 'v', 'f')
    def symbolic_multi_label_nms(g, boxes, scores, iou_threshold):#, max_output_per_class, iou_threshold, score_threshold):
        boxes = unsqueeze(g, boxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        max_output_per_class = g.op('Constant', value_t=torch.tensor([2000], dtype=torch.long))
        iou_threshold = g.op('Constant', value_t=torch.tensor([iou_threshold], dtype=torch.float))
        nms_out = g.op('NonMaxSuppression', boxes, scores, max_output_per_class, iou_threshold)
        return squeeze(g, select(g, nms_out, 1, g.op('Constant', value_t=torch.tensor([2], dtype=torch.long))), 1)

    @parse_args('v', 'v', 'f', 'i', 'i', 'i')
    def symbolic_roi_align(g, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
        batch_indices = _cast_Long(g, squeeze(g, select(g, rois, 1, g.op('Constant', value_t=torch.tensor([0], dtype=torch.long))), 1), False)
        rois = select(g, rois, 1, g.op('Constant', value_t=torch.tensor([1,2,3,4], dtype=torch.long)))
        return g.op('RoiAlign', input, rois, batch_indices, spatial_scale_f=spatial_scale, output_height_i=pooled_height, output_width_i=pooled_width, sampling_ratio_i=sampling_ratio)

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic('DetectionHub::nms', symbolic_multi_label_nms, 10)
    register_custom_op_symbolic('DetectionHub::roi_align_forward', symbolic_roi_align, 10)

register_custom_op()

def postprocess_model(model_path):
    import onnx
    onnx_model = onnx.load(model_path)

    def update_inputs_outputs_dims(model, input_dims, output_dims):
        """
            This function updates the sizes of dimensions of the model's inputs and outputs to the values
            provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
            will be set for that dimension.
        """
        def update_dim(tensor, dim, i, j, dim_param_prefix):
            dim_proto = tensor.type.tensor_type.shape.dim[j]
            if isinstance(dim, int):
                if dim >= 0:
                    dim_proto.dim_value = dim
                else:
                    dim_proto.dim_param = dim_param_prefix + str(i) + '_' + str(j)
            elif isinstance(dim, str):
                dim_proto.dim_param = dim
            else:
                raise ValueError('Only int or str is accepted as dimension value, incorrect type: {}'.format(type(dim)))

        for i, input_dim_arr in enumerate(input_dims):
            for j, dim in enumerate(input_dim_arr):
                update_dim(model.graph.input[i], dim, i, j, 'in_')

        for i, output_dim_arr in enumerate(output_dims):
            for j, dim in enumerate(output_dim_arr):
                update_dim(model.graph.output[i], dim, i, j, 'out_')

        onnx.checker.check_model(model)
        return model

    def remove_unused_floor(model):
        nodes = model.graph.node

        for i, n in enumerate(nodes):
            n.name = str(i)

        floor_nodes = [node for node in nodes if node.op_type=='Floor']

        for f in floor_nodes:
            in_id = f.input[0]
            out_id = f.output[0]
            in_n = [node for node in nodes if node.output == [in_id]][0]
            if in_n.op_type == 'Mul':
                out_n = [node for node in nodes if node.input == [out_id]][0]
                out_n.input[0] = in_n.output[0]
                nodes.remove(f)

        return model
    onnx_model = remove_unused_floor(onnx_model)
    onnx_model = update_inputs_outputs_dims(onnx_model, [[3, 'height', 'width']], [['nbox', 4], ['nbox'], ['nbox']])
    onnx.save(onnx_model, model_path)