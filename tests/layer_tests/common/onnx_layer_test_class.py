# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from common.layer_test_class import CommonLayerTest
from common.layer_utils import BaseInfer

def test_generate_proposals(inputs_dict):
    import torch
    from caffe2.python import core, workspace
    import numpy as np

    scores = inputs_dict['scores']
    im_info = inputs_dict['im_info']
    bbox_deltas = inputs_dict['bbox_deltas']
    anchors = inputs_dict['anchors']

    #scores_gt = np.ones((img_count, A, H, W)).astype(np.float32)
    #bbox_deltas_gt = (
    #    np.linspace(0, 10, num=img_count * 4 * A * H * W)
    #    .reshape((img_count, 4 * A, H, W))
    #    .astype(np.float32)
    #)
    #im_info_gt = np.ones((img_count, 3)).astype(np.float32) / 10
    #anchors_gt = np.ones((A, 4)).astype(np.float32)

    print("======================================")
    print(im_info.shape)
    print(anchors.shape)
    print(bbox_deltas.shape)
    print(scores.shape)
    print("======================================")

    def generate_proposals_ref():
        ref_op = core.CreateOperator(
            "GenerateProposals",
            ["scores", "bbox_deltas", "im_info", "anchors"],
            ["rois", "rois_probs"],
            spatial_scale=2.0,
        )
        workspace.FeedBlob("scores", scores)
        workspace.FeedBlob("bbox_deltas", bbox_deltas)
        workspace.FeedBlob("im_info", im_info)
        workspace.FeedBlob("anchors", anchors)
        workspace.RunOperatorOnce(ref_op)
        return workspace.FetchBlob("rois"), workspace.FetchBlob("rois_probs")

    rois, rois_probs = generate_proposals_ref()
    rois = torch.tensor(rois)
    rois_probs = torch.tensor(rois_probs)
    print(rois)
    print(rois_probs)

def save_to_onnx(onnx_model, path_to_saved_onnx_model):
    import onnx
    path = os.path.join(path_to_saved_onnx_model, 'model.onnx')
    onnx.save(onnx_model, path)
    assert os.path.isfile(path), "model.onnx haven't been saved here: {}".format(path_to_saved_onnx_model)
    return path


class Caffe2OnnxLayerTest(CommonLayerTest):
    def produce_model_path(self, framework_model, save_path):
        return save_to_onnx(framework_model, save_path)

    def get_framework_results(self, inputs_dict, model_path):
        # Evaluate model via Caffe2 and IE
        # Load the ONNX model
        print("====================Starting framework inference")
        import onnx
        model = onnx.load(model_path)
        print("====================Starting framework inference")
        # Run the ONNX model with Caffe2
        test_generate_proposals(inputs_dict)
        #import caffe2.python.onnx.backend
        #caffe2_res = caffe2.python.onnx.backend.run_model(model, inputs_dict)
        print("====================Starting framework inference")
        res = dict()
        #for field in caffe2_res._fields:
        #    res[field] = caffe2_res[field]
        return res


class OnnxRuntimeInfer(BaseInfer):
    def __init__(self, net):
        super().__init__('OnnxRuntime')
        self.net = net

    def fw_infer(self, input_data):
        import onnxruntime as rt

        sess = rt.InferenceSession(self.net)
        out = sess.run(None, input_data)
        result = dict()
        for i, output in enumerate(sess.get_outputs()):
            result[output.name] = out[i]

        if "sess" in locals():
            del sess

        return result


class OnnxRuntimeLayerTest(CommonLayerTest):
    def produce_model_path(self, framework_model, save_path):
        return save_to_onnx(framework_model, save_path)

    def get_framework_results(self, inputs_dict, model_path):
        ort = OnnxRuntimeInfer(net=model_path)
        res = ort.infer(input_data=inputs_dict)
        return res
