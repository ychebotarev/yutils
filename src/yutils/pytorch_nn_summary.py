import numpy as np

import torch
import torch.nn as nn

#heavily inspired by the implementation of summary for pytorch
class PytorchModelSummaryCollector:
    def __init__(self, input_size, batch_size):
        self.summaries = []
        self.input_size = input_size
        self.batch_size = batch_size
        self.was_batch_norm = False
        self.total_output = 0
        self.trainable_params = 0
        self.total_params = 0
        
    def __call__(self, module, module_in, module_out):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(self.summaries)
        
        layer_name = f"{class_name}-{module_idx + 1}"
        if layer_name.startswith('BatchNorm'):
            self.was_batch_norm = True
        
        summary={}
        summary['name'] = layer_name
        summary["input_shape"] = list(module_in[0].size())
        if isinstance(module_out, (list, tuple)):
            summary["output_shape"] = [
                [-1] + list(o.size())[1:] for o in module_out
            ]
        else:
            summary["output_shape"] = list(module_out.size())
        
        self.total_output += np.prod(summary["output_shape"])
        params = 0
        summary["trainable"] = False
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += torch.prod(torch.LongTensor(list(module.weight.size())))
            summary["trainable"] = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += torch.prod(torch.LongTensor(list(module.bias.size())))
        summary["params"] = params
        self.total_params += params
        if summary["trainable"] == True:
            self.trainable_params += params
        self.summaries.append(summary)

    def __repr__(self):
        summary_str =  "================================================================\n"
        line_new = "{:>20}{:>20}{:>20}{:>10}".format("Layer (type)", "Input Shape", "Output Shape", "Param #")
        summary_str += line_new + "\n"
        summary_str += "================================================================\n"
        for summary in self.summaries:
            output_shape=summary["output_shape"]
            input_shape=summary["input_shape"]
            if self.was_batch_norm == False:
                output_shape.pop(0)
                input_shape.pop(0)
            str_output_shape = str(output_shape)
            str_input_shape = str(input_shape)
            str_params = f'{summary["params"]}'
            line_new = f"{summary['name']:>20}{str_input_shape:>20}{str_output_shape:>20}{str_params:>10}\n"
            summary_str += line_new

        batch_size=1 if self.batch_size == -1 else self.batch_size
        # assume 4 bytes/number (float on cuda).
        total_input_size = np.prod(self.input_size) * batch_size * 4. / (1024 ** 2.)
        # x2 for gradients
        total_output_size = 2. * self.total_output * 4. / (1024 ** 2.)
        total_params_size = self.total_params * 4. / (1024 ** 2.)
        total_size = total_params_size + total_output_size + total_input_size        
        

        summary_str += "================================================================\n"
        summary_str += f"Total params: {self.total_params:,}\n"
        summary_str += f"Trainable params: {self.trainable_params:,}\n"
        summary_str += f"Non-trainable params: {self.total_params - self.trainable_params:,}\n"
        summary_str += "----------------------------------------------------------------\n"
        summary_str += f"Input size (MB): {total_input_size:.2f}\n"
        summary_str += f"Forward/backward pass size (MB): {total_output_size:.2f}\n"
        summary_str += f"Params size (MB): {total_params_size:.2f}\n"
        summary_str += f"Estimated Total Size (MB): {total_size:.2f}\n"
        summary_str += "================================================================\n"            
        return summary_str

class PytorchModelHookRegister:
    def __init__(self, collector) -> None:
        self.hooks = []
        self.collector = collector
    
    def __call__(self, module):
        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            self.hooks.append(module.register_forward_hook(self.collector))        

class PytorchModelSummary:
    def __init__(self, model, input_size, batch_size=-1):
        self.model = model
        if isinstance(input_size, tuple):
            input_size = [input_size]
        if isinstance(input_size, int):
            input_size=[[input_size]]
        if not isinstance(input_size, list):
            raise TypeError(f'Unsupported input size type:{type(input_size)}')
        if len(input_size)<1:
            raise ValueError(f'Empty input size')
        if not isinstance(input_size[0], list) and not isinstance(input_size[0], tuple):
            raise TypeError(f'Unsupported input size type:{type(input_size)}')

        self.input_size = input_size
        self.batch_size = batch_size

    def register_hooks(self, model, collector):
        hook_handles = []
        for module in model.children():        
            if isinstance(module, torch.nn.modules.container.Sequential):
                sub_hooks = self.register_hooks(module, collector)
                hook_handles.extend(sub_hooks)
            else:
                handle = module.register_forward_hook(collector)
                hook_handles.append(handle)
        return hook_handles

    def get_summary(self):
        collector = PytorchModelSummaryCollector(self.input_size, self.batch_size)
        hook_handles = self.register_hooks(self.model, collector)

        device=next(self.model.parameters()).device
        dtype = torch.FloatTensor if device.type == 'cpu' else torch.cuda.FloatTensor
        dtypes = [dtype]*len(self.input_size)
        x = [torch.rand(2, *in_size).type(dtype).to(device=device)
             for in_size, dtype in zip(self.input_size, dtypes)]
        summary_str = ''
        try:
            self.model(*x)
        except Exception as e:
            summary_str += 'Exception while running model summary:\n'
            summary_str += str(e)
            summary_str += '\n Data collected so far:\n'
        finally:
            summary_str += str(collector)
            for handle in hook_handles:
                handle.remove()
        
        return summary_str, collector.summaries

def nn_summary(model, input_size, batch_size=-1):
    pms = PytorchModelSummary(model, input_size, batch_size)
    return pms.get_summary()
