#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy

# adds the input nodes and returns the descriptor
def AddInputLayer(config_lines, feat_dim, splice_indexes=[0], ivector_dim=0):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    output_dim = 0
    components.append('input-node name=input dim=' + str(feat_dim))
    list = [('Offset(input, {0})'.format(n) if n != 0 else 'input') for n in splice_indexes]
    output_dim += len(splice_indexes) * feat_dim
    if ivector_dim > 0:
        components.append('input-node name=ivector dim=' + str(ivector_dim))
        list.append('ReplaceIndex(ivector, t, 0)')
        output_dim += ivector_dim
    splice_descriptor = "Append({0})".format(", ".join(list))
    print(splice_descriptor)
    return {'descriptor': splice_descriptor,
            'dimension': output_dim}

def AddLdaLayer(config_lines, name, input, lda_file):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={0}_lda type=FixedAffineComponent matrix={1}'.format(name, lda_file))
    component_nodes.append('component-node name={0}_lda component={0}_lda input={1}'.format(name, input['descriptor']))

    return {'descriptor':  '{0}_lda'.format(name),
            'dimension': input['dimension']}

def AddAffineLayer(config_lines, name, input, output_dim, ng_affine_options = ""):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))

    return {'descriptor':  '{0}_affine'.format(name),
            'dimension': output_dim}

def AddAffRelNormLayer(config_lines, name, input, output_dim, ng_affine_options = ""):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1}".format(name, output_dim))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1}".format(name, output_dim))

    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': output_dim}



def AddSoftmaxLayer(config_lines, name, input):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_log_softmax type=LogSoftmaxComponent dim={1}".format(name, input['dimension']))
    component_nodes.append("component-node name={0}_log_softmax component={0}_log_softmax input={1}".format(name, input['descriptor']))

    return {'descriptor':  '{0}_log_softmax'.format(name),
            'dimension': input['dimension']}


def AddOutputNode(config_lines, input, label_delay=None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    if label_delay is None:
        component_nodes.append('output-node name=output input={0}'.format(input['descriptor']))
    else:
        component_nodes.append('output-node name=output input=Offset({0},{1})'.format(input['descriptor'], label_delay))

def AddFinalLayer(config_lines, input, output_dim, ng_affine_options = "", label_delay=None):
    prev_layer_output = AddAffineLayer(config_lines, "Final", input, output_dim, ng_affine_options)
    prev_layer_output = AddSoftmaxLayer(config_lines, "Final", prev_layer_output)
    AddOutputNode(config_lines, prev_layer_output, label_delay)


def AddLstmLayer(config_lines,
                 name, input, cell_dim,
                 recurrent_projection_dim = 0,
                 non_recurrent_projection_dim = 0,
                 clipping_threshold = 1.0,
                 norm_based_clipping = "false",
                 ng_per_element_scale_options = "",
                 ng_affine_options = "",
                 lstm_delay = -1):
    assert(recurrent_projection_dim >= 0 and non_recurrent_projection_dim >= 0)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    input_descriptor = input['descriptor']
    input_dim = input['dimension']
    name = name.strip()

    if (recurrent_projection_dim == 0):
        add_recurrent_projection = False
        recurrent_projection_dim = cell_dim
        recurrent_connection = "m_t"
    else:
        add_recurrent_projection = True
        recurrent_connection = "r_t"
    if (non_recurrent_projection_dim == 0):
        add_non_recurrent_projection = False
    else:
        add_non_recurrent_projection = True

    # Natural gradient per element scale parameters
    ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
    # Parameter Definitions W*(* replaced by - to have valid names)
    components.append("# Input gate control : W_i* matrices")
    components.append("component name={0}_W_i-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_ic type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Forget gate control : W_f* matrices")
    components.append("component name={0}_W_f-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_fc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("#  Output gate control : W_o* matrices")
    components.append("component name={0}_W_o-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_oc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Cell input matrices : W_c* matrices")
    components.append("component name={0}_W_c-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))


    components.append("# Defining the non-linearities")
    components.append("component name={0}_i type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_f type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_o type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_g type=TanhComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_h type=TanhComponent dim={1}".format(name, cell_dim))

    components.append("# Defining the cell computations")
    components.append("component name={0}_c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))

    # c1_t and c2_t defined below
    component_nodes.append("component-node name={0}_c_t component={0}_c input=Sum({0}_c1_t, {0}_c2_t)".format(name))
    c_tminus1_descriptor = "IfDefined(Offset({0}_c_t, {1}))".format(name, lstm_delay)

    component_nodes.append("# i_t")
    component_nodes.append("component-node name={0}_i1 component={0}_W_i-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_i2 component={0}_w_ic  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_i_t component={0}_i input=Sum({0}_i1, {0}_i2)".format(name))

    component_nodes.append("# f_t")
    component_nodes.append("component-node name={0}_f1 component={0}_W_f-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_f2 component={0}_w_fc  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_f_t component={0}_f input=Sum({0}_f1,{0}_f2)".format(name))

    component_nodes.append("# o_t")
    component_nodes.append("component-node name={0}_o1 component={0}_W_o-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_o2 component={0}_w_oc input={0}_c_t".format(name))
    component_nodes.append("component-node name={0}_o_t component={0}_o input=Sum({0}_o1, {0}_o2)".format(name))

    component_nodes.append("# h_t")
    component_nodes.append("component-node name={0}_h_t component={0}_h input={0}_c_t".format(name))

    component_nodes.append("# g_t")
    component_nodes.append("component-node name={0}_g1 component={0}_W_c-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_g_t component={0}_g input={0}_g1".format(name))

    component_nodes.append("# parts of c_t")
    component_nodes.append("component-node name={0}_c1_t component={0}_c1  input=Append({0}_f_t, {1})".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_c2_t component={0}_c2 input=Append({0}_i_t, {0}_g_t)".format(name))

    component_nodes.append("# m_t")
    component_nodes.append("component-node name={0}_m_t component={0}_m input=Append({0}_o_t, {0}_h_t)".format(name))

    # add the recurrent connections
    if (add_recurrent_projection and add_non_recurrent_projection):
        components.append("# projection matrices : Wrm and Wpm")
        components.append("component name={0}_W-m type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim + non_recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t and p_t")
        component_nodes.append("component-node name={0}_rp_t component={0}_W-m input={0}_m_t".format(name))
        component_nodes.append("dim-range-node name={0}_r_t_preclip input-node={0}_rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_rp_t'.format(name)
        output_dim = recurrent_projection_dim + non_recurrent_projection_dim

    elif add_recurrent_projection:
        components.append("# projection matrices : Wrm")
        components.append("component name={0}_Wrm type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t")
        component_nodes.append("component-node name={0}_r_t_preclip component={0}_Wrm input={0}_m_t".format(name))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = recurrent_projection_dim

    else:
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_m_t".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = cell_dim

    return {
            'descriptor': output_descriptor,
            'dimension':output_dim
           }

def AddClstmLayer(config_lines,
                 name, input, cell_dim,
                 recurrent_projection_dim = 0,
                 non_recurrent_projection_dim = 0,
                 clipping_threshold = 1.0,
                 norm_based_clipping = "false",
                 ng_per_element_scale_options = "",
                 ng_affine_options = "",
                 lstm_delay = -1,
                 rates = [1]):
    assert(recurrent_projection_dim >= 0 and non_recurrent_projection_dim >= 0)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    input_descriptor = input['descriptor']
    input_dim = input['dimension']
    name = name.strip()

    if (recurrent_projection_dim == 0):
        add_recurrent_projection = False
        recurrent_projection_dim = cell_dim
        recurrent_connection = "m_t"
    else:
        add_recurrent_projection = True
        recurrent_connection = "r_t"
    if (non_recurrent_projection_dim == 0):
        add_non_recurrent_projection = False
    else:
        add_non_recurrent_projection = True

    # Natural gradient per element scale parameters
    ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
    # Parameter Definitions W*(* replaced by - to have valid names)
    components.append("# Input gate control : W_i* matrices")
    components.append("component name={0}_W_i-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_ic type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Forget gate control : W_f* matrices")
    components.append("component name={0}_W_f-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_fc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("#  Output gate control : W_o* matrices")
    components.append("component name={0}_W_o-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_oc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Cell input matrices : W_c* matrices")
    components.append("component name={0}_W_c-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))

    components.append("# Defining the non-linearities")
    components.append("component name={0}_i type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_f type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_o type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_g type=TanhComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_h type=TanhComponent dim={1}".format(name, cell_dim))

    components.append("# Defining the cell computations")
    components.append("component name={0}_c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))

    # c1_t and c2_t defined below
    component_nodes.append("component-node name={0}_c_t component={0}_c input=Sum({0}_c1_t, {0}_c2_t)".format(name))
    c_tminus1_descriptor = "IfDefined(Offset({0}_c_t, {1}))".format(name, lstm_delay)

    component_nodes.append("# i_t")
    component_nodes.append("component-node name={0}_i1 component={0}_W_i-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_i2 component={0}_w_ic  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_i_t component={0}_i input=Sum({0}_i1, {0}_i2)".format(name))

    component_nodes.append("# f_t")
    component_nodes.append("component-node name={0}_f1 component={0}_W_f-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_f2 component={0}_w_fc  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_f_t component={0}_f input=Sum({0}_f1,{0}_f2)".format(name))

    component_nodes.append("# o_t")
    component_nodes.append("component-node name={0}_o1 component={0}_W_o-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_o2 component={0}_w_oc input={0}_c_t".format(name))
    component_nodes.append("component-node name={0}_o_t component={0}_o input=Sum({0}_o1, {0}_o2)".format(name))

    component_nodes.append("# h_t")
    component_nodes.append("component-node name={0}_h_t component={0}_h input={0}_c_t".format(name))

    component_nodes.append("# g_t")
    component_nodes.append("component-node name={0}_g1 component={0}_W_c-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_g_t component={0}_g input={0}_g1".format(name))

    component_nodes.append("# parts of c_t")
    component_nodes.append("component-node name={0}_c1_t component={0}_c1  input=Append({0}_f_t, {1})".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_c2_t component={0}_c2 input=Append({0}_i_t, {0}_g_t)".format(name))

    component_nodes.append("# m_t")
    component_nodes.append("component-node name={0}_m_t component={0}_m input=Append({0}_o_t, {0}_h_t)".format(name))

    # add the recurrent connections
    if (add_recurrent_projection and add_non_recurrent_projection):
        components.append("# projection matrices : Wrm and Wpm")
        components.append("component name={0}_W-m type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim + non_recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t and p_t")
        component_nodes.append("component-node name={0}_rp_t component={0}_W-m input={0}_m_t".format(name))
        component_nodes.append("dim-range-node name={0}_r_t_preclip input-node={0}_rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_rp_t'.format(name)
        output_dim = recurrent_projection_dim + non_recurrent_projection_dim

    elif add_recurrent_projection:
        components.append("# projection matrices : Wrm")
        components.append("component name={0}_Wrm type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t")
        component_nodes.append("component-node name={0}_r_t_preclip component={0}_Wrm input={0}_m_t".format(name))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = recurrent_projection_dim

    else:
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_m_t".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = cell_dim

    return {
            'descriptor': output_descriptor,
            'dimension':output_dim
           }

'''
Mathematical equations that specify one BLSTM layer:
input: x(t), output: y(t), cell: c(t), recurrent connection: r(t), lstm_delay: -d
i_forward(t) = sigmoid(Wix * x(t) + Wir * r_forward(t-d) + wic * c_forward(t-1) + bi)
i_backward(t) = sigmoid(Wix * x(t) + Wir * r_backward(t+d) + wic * c_backward(t+1) + bi)
f_forward(t) = sigmoid(Wfx * x(t) + Wfr * r_forward(t-d) + wfc * c_backward(t-1) + bf)
f_backward(t) = sigmoid(Wfx * x(t) + Wfr * r_backward(t+d) + wfc * c_backward(t+1) + bf)
c_forward(t) = {f_forward(t) .* c(t-1)} + {i_forward(t) .* tanh(Wcx * x(t) + Wcr * r_forward(t-d) + bc)}
c_backward(t) = {f_backward(t) .* c(t+1)} + {i_backward(t) .* tanh(Wcx * x(t) + Wcr * r_backward(t+d) + bc)}
o_forward(t) = sigmoid(Wox * x(t) + Wor * r_forward(t-d) + woc * c_forward(t) + bo)
o_backward(t) = sigmoid(Wox * x(t) + Wor * r_backward(t+d) + woc * c_backward(t) + bo)
m_forward(t) = o_forward(t) .* tanh(c_forward(t))
m_backward(t) = o_backward(t) .* tanh(c_backward(t))
r_forward(t) = Wrm * m_forward(t) + br
r_backward(t) = Wrm * m_backward(t) + br
p_forward(t) = Wpm * m_forward(t) + bp
p_backward(t) = Wpm * m_backward(t) + bp
y(t) = [r_forward(t); p_forward(t); r_backward(t); p_backward(t)]

Equations at implementation level are generally equivalent to the above equations (except some small variations):
i1_forward(t) = W_i-xr * [x(t); r_forward(t-d)] + b_i-xr
i2_forward(t) = w_ic * c_forward(t-1) + b_ic
i_forward(t) = i1_forward(t) + i2_forward(t)
i1_backward(t) = W_i-xr * [x(t); r_backward(t+d)] + b_i-xr
i2_backward(t) = w_ic * c_backward(t+1) + b_ic
i_backward(t) = i1_backward(t) + i2_backward(t)

f1_forward(t) = W_f-xr * [x(t); r_forward(t-d)] + b_f-xr
f2_forward(t) = w_fc * c_forward(t-1) + b_fc
f_forward(t) = f1_forward(t) + f2_forward(t)
f1_backward(t) = W_f-xr * [x(t); r_backward(t+d)] + b_f-xr
f2_backward(t) = w_fc * c_backward(t+1) + b_fc
f_backward(t) = f1_backward(t) + f2_backward(t)

g1_forward(t) = W_c-xr * [x(t); r_forward(t-d)] + b_c-xr
g_forward(t) = tanh(g1_forward(t))
g1_backward(t) = W_c-xr * [x(t); r_backward(t+d)] + b_c-xr
g_backward(t) = tanh(g1_backward(t))

c1_forward(t) = f_forward(t) .* c_forward(t-1)
c2_forward(t) = i_forward(t) .* g_forward(t)
c_forward(t) = c1_forward(t) + c2_forward(t)
c1_backward(t) = f_backward(t) .* c_backward(t+1)
c2_backward(t) = i_backward(t) .* g_backward(t)
c_backward(t) = c1_backward(t) + c2_backward(t)

h_forward(t) = tanh(c_forward(t))
h_backward(t) = tanh(c_backward(t))

o1_forward(t) = W_o-xr * [x(t); r_forward(t-d)] + b_o-xr
o2_forward(t) = w_oc * c_forward(t) + b_oc
o_forward(t) = o1_forward(t) + o2_forward(t)
o1_backward(t) = W_o-xr * [x(t); r_backward(t+d)] + b_o-xr
o2_backward(t) = w_oc * c_backward(t) + b_oc
o_backward(t) = o1_backward(t) + o2_backward(t)

m_forward(t) = o_forward(t) .* h_forward(t)
m_backward(t) = o_backward(t) .* h_backward(t)

rp_forward(t) = W-m * m_forward(t) + b-m
r_forward(t) = clip(dim-range(rp_forward(t)))
rp_backward(t) = W-m * m_backward(t) + b-m
r_backward(t) = clip(dim-range(rp_backward(t)))

y(t) = [rp_forward(t); rp_backward(t)]
'''
def AddBlstmLayer(config_lines,
                 name, input, cell_dim,
                 recurrent_projection_dim = 0,
                 non_recurrent_projection_dim = 0,
                 clipping_threshold = 1.0,
                 norm_based_clipping = "false",
                 ng_per_element_scale_options = "",
                 ng_affine_options = "",
                 lstm_delay = -1):
    assert(recurrent_projection_dim >= 0 and non_recurrent_projection_dim >= 0)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    input_descriptor = input['descriptor']
    if (input_descriptor.startswith('Append')):
        input_descriptor = input_descriptor[7:-1]
    input_dim = input['dimension']
    name = name.strip()

    if (recurrent_projection_dim == 0):
        add_recurrent_projection = False
        recurrent_projection_dim = cell_dim
        forward_recurrent_connection = "fw_m_t"
	backward_recurrent_connection = "bw_m_t"
    else:
        add_recurrent_projection = True
        forward_recurrent_connection = "fw_r_t"
	backward_recurrent_connection = "bw_r_t"
    if (non_recurrent_projection_dim == 0):
        add_non_recurrent_projection = False
    else:
        add_non_recurrent_projection = True

    # Natural gradient per element scale parameters
    ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
    # Parameter Definitions W*(* replaced by - to have valid names)
    components.append("# Input gate control : W_i* matrices")
    components.append("component name={0}_W_i-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_ic type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Forget gate control : W_f* matrices")
    components.append("component name={0}_W_f-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_fc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("#  Output gate control : W_o* matrices")
    components.append("component name={0}_W_o-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_oc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Cell input matrices : W_c* matrices")
    components.append("component name={0}_W_c-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))


    components.append("# Defining the non-linearities")
    components.append("component name={0}_i type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_f type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_o type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_g type=TanhComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_h type=TanhComponent dim={1}".format(name, cell_dim))

    components.append("# Defining the cell computations")
    components.append("component name={0}_c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))

    # c1_t_{forward|backward} and c2_t_{forward|backward} defined below
    # c_t_forward = c1_t_forward + c2_t_forward
    component_nodes.append("component-node name={0}_fw_c_t component={0}_c input=Sum({0}_fw_c1_t, {0}_fw_c2_t)".format(name))
    # c_t_backward = c1_t_backward + c2_t_backward
    component_nodes.append("component-node name={0}_bw_c_t component={0}_c input=Sum({0}_bw_c1_t, {0}_bw_c2_t)".format(name))
    # c_t-1_forward
    c_tminus1_descriptor = "IfDefined(Offset({0}_fw_c_t, {1}))".format(name, lstm_delay)
    # c_t-1_backward
    c_tplus1_descriptor = "IfDefined(Offset({0}_bw_c_t, {1}))".format(name, -lstm_delay)

    component_nodes.append("# i_t")
    # i_t_forward = sigmoid(W_i * [input_t; r_t-d_forward; c_t-1_forward] + b_i)
    component_nodes.append("component-node name={0}_fw_i1 component={0}_W_i-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, forward_recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_fw_i2 component={0}_w_ic  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_fw_i_t component={0}_i input=Sum({0}_fw_i1, {0}_fw_i2)".format(name))
    # i_t_backward = sigmoid(W_i * [input_t; r_t+d_backward; c_t+1_backward] + b_i)
    component_nodes.append("component-node name={0}_bw_i1 component={0}_W_i-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, backward_recurrent_connection, -lstm_delay))
    component_nodes.append("component-node name={0}_bw_i2 component={0}_w_ic  input={1}".format(name, c_tplus1_descriptor))
    component_nodes.append("component-node name={0}_bw_i_t component={0}_i input=Sum({0}_bw_i1, {0}_bw_i2)".format(name))

    component_nodes.append("# f_t")
    # f_t_forward = sigmoid(W_f * [input_t; r_t-d_forward; c_t-1_forward] + b_f)
    component_nodes.append("component-node name={0}_fw_f1 component={0}_W_f-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, forward_recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_fw_f2 component={0}_w_fc  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_fw_f_t component={0}_f input=Sum({0}_fw_f1,{0}_fw_f2)".format(name))
    # f_t_backward = sigmoid(W_f * [input_t; r_t+d_backward; c_t+1_backward] + b_f)
    component_nodes.append("component-node name={0}_bw_f1 component={0}_W_f-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, backward_recurrent_connection, -lstm_delay))
    component_nodes.append("component-node name={0}_bw_f2 component={0}_w_fc  input={1}".format(name, c_tplus1_descriptor))
    component_nodes.append("component-node name={0}_bw_f_t component={0}_f input=Sum({0}_bw_f1,{0}_bw_f2)".format(name))

    component_nodes.append("# o_t")
    # o_t_forward = sigmoid(W_o * [input_t; r_t-d_forward; c_t_forward] + b_o)
    component_nodes.append("component-node name={0}_fw_o1 component={0}_W_o-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, forward_recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_fw_o2 component={0}_w_oc input={0}_fw_c_t".format(name))
    component_nodes.append("component-node name={0}_fw_o_t component={0}_o input=Sum({0}_fw_o1, {0}_fw_o2)".format(name))
    # o_t_backward = sigmoid(W_o * [input_t; r_t+d_backward; c_t_backward] + b_o)
    component_nodes.append("component-node name={0}_bw_o1 component={0}_W_o-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, backward_recurrent_connection, -lstm_delay))
    component_nodes.append("component-node name={0}_bw_o2 component={0}_w_oc input={0}_bw_c_t".format(name))
    component_nodes.append("component-node name={0}_bw_o_t component={0}_o input=Sum({0}_bw_o1, {0}_bw_o2)".format(name))

    component_nodes.append("# h_t")
    # h_t_forward = tanh(c_t_forward)
    component_nodes.append("component-node name={0}_fw_h_t component={0}_h input={0}_fw_c_t".format(name))
    # h_t_backward = tanh(c_t_backward)
    component_nodes.append("component-node name={0}_bw_h_t component={0}_h input={0}_bw_c_t".format(name))

    component_nodes.append("# g_t")
    # g_t_forward = tanh(W_g * [input_t; r_t-d_forward] + b_g)
    component_nodes.append("component-node name={0}_fw_g1 component={0}_W_c-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, forward_recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_fw_g_t component={0}_g input={0}_fw_g1".format(name))
    # g_t_backward = tanh(W_g * [input_t; r_t+d_backward] + b_g)
    component_nodes.append("component-node name={0}_bw_g1 component={0}_W_c-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, backward_recurrent_connection, -lstm_delay))
    component_nodes.append("component-node name={0}_bw_g_t component={0}_g input={0}_bw_g1".format(name))

    component_nodes.append("# parts of c_t")
    # c1_t_forward = f_t_forward .* c_t-1_forward
    component_nodes.append("component-node name={0}_fw_c1_t component={0}_c1  input=Append({0}_fw_f_t, {1})".format(name, c_tminus1_descriptor))
    # c2_t_forward = i_t_forward .* g_t_forward
    component_nodes.append("component-node name={0}_fw_c2_t component={0}_c2 input=Append({0}_fw_i_t, {0}_fw_g_t)".format(name))
    # c1_t_backward = f_t_backward .* c_t+1_backward
    component_nodes.append("component-node name={0}_bw_c1_t component={0}_c1  input=Append({0}_bw_f_t, {1})".format(name, c_tplus1_descriptor))
    # c2_t_backward = i_t_backward .* g_t_backward
    component_nodes.append("component-node name={0}_bw_c2_t component={0}_c2 input=Append({0}_bw_i_t, {0}_bw_g_t)".format(name))

    component_nodes.append("# m_t")
    # m_t_forward = o_t_forward .* h_t_forward
    component_nodes.append("component-node name={0}_fw_m_t component={0}_m input=Append({0}_fw_o_t, {0}_fw_h_t)".format(name))
    # m_t_backward = o_t_backward .* h_t_backward
    component_nodes.append("component-node name={0}_bw_m_t component={0}_m input=Append({0}_bw_o_t, {0}_bw_h_t)".format(name))

    # add the recurrent connections
    if (add_recurrent_projection and add_non_recurrent_projection):
        components.append("# projection matrices : Wrm and Wpm")
        components.append("component name={0}_W-m type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim + non_recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t and p_t")
	# rp_t_forward = W * m_t_forward + b; r_t_preclip_forward = dim-range(rp_t_forward); r_t_forward = clip(r_t_preclip_forward)
        component_nodes.append("component-node name={0}_fw_rp_t component={0}_W-m input={0}_fw_m_t".format(name))
        component_nodes.append("dim-range-node name={0}_fw_r_t_preclip input-node={0}_fw_rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        component_nodes.append("component-node name={0}_fw_r_t component={0}_r input={0}_fw_r_t_preclip".format(name))
	# rp_t_backward = W * m_t_backward + b; r_t_preclip_backward = dim-range(rp_t_backward); r_t_backward = clip(r_t_preclip_backward)
	component_nodes.append("component-node name={0}_bw_rp_t component={0}_W-m input={0}_bw_m_t".format(name))
        component_nodes.append("dim-range-node name={0}_bw_r_t_preclip input-node={0}_bw_rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        component_nodes.append("component-node name={0}_bw_r_t component={0}_r input={0}_bw_r_t_preclip".format(name))
 
        # output_t = [rp_t_forward; rp_t_backward]
	output_descriptor = 'Append({0}_fw_rp_t, {0}_bw_rp_t)'.format(name)
        output_dim = (recurrent_projection_dim + non_recurrent_projection_dim) * 2

    elif add_recurrent_projection:
        components.append("# projection matrices : Wrm")
        components.append("component name={0}_Wrm type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t")
	# r_t_preclip_forward = W * m_t_forward + b; r_t_forward = clip(r_t_preclip_forward)
        component_nodes.append("component-node name={0}_fw_r_t_preclip component={0}_Wrm input={0}_fw_m_t".format(name))
        component_nodes.append("component-node name={0}_fw_r_t component={0}_r input={0}_fw_r_t_preclip".format(name))
	# r_t_preclip_backward = W * m_t_backward + b; r_t_backward = clip(r_t_preclip_backward)
        component_nodes.append("component-node name={0}_bw_r_t_preclip component={0}_Wrm input={0}_bw_m_t".format(name))
        component_nodes.append("component-node name={0}_bw_r_t component={0}_r input={0}_bw_r_t_preclip".format(name))
 
        # output_t = [r_t_forward; r_t_backward]
        output_descriptor = 'Append({0}_fw_r_t, {0}_bw_r_t)'.format(name)
        output_dim = recurrent_projection_dim * 2

    else:
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))
	# r_t_forward = clip(m_t_forward)
        component_nodes.append("component-node name={0}_fw_r_t component={0}_r input={0}_fw_m_t".format(name))
	# r_t_backward = clip(m_t_backward)
        component_nodes.append("component-node name={0}_bw_r_t component={0}_r input={0}_bw_m_t".format(name))
 
        # output_t = [r_t_forward; r_t_backward]
        output_descriptor = 'Append({0}_fw_r_t, {0}_bw_r_t)'.format(name)
        output_dim = cell_dim * 2

    return {
            'descriptor': output_descriptor,
            'dimension':output_dim
           }


