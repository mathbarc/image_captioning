from typing import Dict
import torch
import math

from torch.optim.optimizer import Optimizer


class RampUpScheduler:
    def __init__(self, optimizer:torch.optim.Optimizer, lr:float, rampup_period:int, lr_minimum:float = 1e-8, power:int = 4):
        self._optimizer = optimizer
        
        self._lr_base = lr
        self._lr_rampup_minimum = lr_minimum
        self._lr_rampup_magnitude = lr-lr_minimum
        self._current_step = 0
        
        self._rampup_period = rampup_period
        self._power = power
    
    def step(self):
        
        lr = self.get_last_lr()
            
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        
        self._current_step+=1
            
    def get_last_lr(self):
        
        if self._current_step >= self._rampup_period:
            lr = self._computer_lr()
            
        elif self._current_step <= self._rampup_period:
            lr = self._lr_rampup_minimum + (self._lr_rampup_magnitude * pow((self._current_step / self._rampup_period), self._power))
            
        return lr
    
    def _computer_lr(self):
        return self._lr_base
 
class RampUpExponentialDecayScheduler(RampUpScheduler):
    def __init__(self, optimizer:torch.optim.Optimizer, lr_base:float, lr_final:float, n_steps:int, rampup_period:int, lr_minimum:float = 1e-8, power:int=4):
        super().__init__(optimizer, lr_base, rampup_period, lr_minimum, power)
        
        self._decay_period = n_steps-rampup_period
        self._decay_amplitude = self._lr_base - lr_final
        self._lr_final = lr_final
    
    def _computer_lr(self):
        return self._lr_final + self._decay_amplitude * pow(((self._decay_period-(self._current_step-self._rampup_period)) / self._decay_period), self._power)
    
class RampUpCosineDecayScheduler(RampUpScheduler):
    def __init__(self, optimizer:torch.optim.Optimizer, lr_base:float, lr_final:float, n_steps:int, rampup_period:int, lr_minimum:float = 1e-8, power:int=4):
        super().__init__(optimizer, lr_base, rampup_period, lr_minimum, power)
        
        self._decay_period = n_steps-rampup_period
        self._decay_amplitude = self._lr_base - lr_final
        self._lr_final = lr_final

    def _computer_lr(self):
        return self._lr_final + self._decay_amplitude * math.sin((math.pi/2)*((self._current_step-self._rampup_period)/self._decay_period)+(math.pi/2))

class RampUpLogisticDecayScheduler(RampUpScheduler):
    def __init__(self, optimizer:torch.optim.Optimizer, lr_base:float, lr_final:float, n_steps:int, rampup_period:int, lr_minimum:float = 1e-8, power:int=4):
        super().__init__(optimizer, lr_base, rampup_period, lr_minimum, power)
        
        self._decay_period = n_steps-rampup_period
        self._decay_amplitude = self._lr_base - lr_final
        self._lr_final = lr_final
        
         
    @staticmethod
    def _sigmoid(x):
        return 1/(1+(math.exp(-x)))        

    def _computer_lr(self):
        expoent = ((self._decay_period-(self._current_step-self._rampup_period))/self._decay_period)*12 - 7
        lr = self._lr_final + self._decay_amplitude*self._sigmoid(expoent)
        return lr

class RampUpCosineAnnealingScheduler(RampUpScheduler):
    def __init__(self, optimizer:torch.optim.Optimizer, lr_base:float, lr_final:float, rampup_period:int, cosine_period:int, cosine_period_inc:float, lr_minimum:float = 1e-8, power:int=4):
        super().__init__(optimizer, lr_base, rampup_period, lr_minimum, power)
        
        self._cosine_period = cosine_period
        self._cosine_period_inc = cosine_period_inc
        
        self._decay_amplitude = self._lr_base - lr_final
        self._lr_final = lr_final
        
        self._cosine_interval_start = self._rampup_period
    
            
    def _computer_lr(self):
        pos_in_interval = (self._current_step-self._cosine_interval_start)
        lr = self._lr_final + self._decay_amplitude * ((math.cos((math.pi/2)*(pos_in_interval/self._cosine_period))))
        if pos_in_interval == self._cosine_period:
            self._cosine_interval_start = self._current_step
            self._cosine_period *= self._cosine_period_inc
        return lr

class RampUpSteps(RampUpScheduler):
    
    def __init__(self, optimizer: Optimizer, lrs: Dict[int,float], rampup_period: int, lr_minimum: float = 1e-8, power: int = 4):
        self.lrs = lrs
        lr = lrs[min(lrs.keys())]
        super().__init__(optimizer, lr, rampup_period, lr_minimum, power)
        
    def _computer_lr(self):
        
        keys = list(self.lrs.keys())
        
        i=0
        while i < len(keys):
            if keys[i]>self._current_step:
                break
            i+=1
        
        if i == len(keys):
            i-=1
        
        return self.lrs[keys[i]]


if __name__=="__main__":
    import matplotlib
    import matplotlib.pyplot

    lr = 1e-3
    epochs = 1000
    lr_ramp_down = 1000
    steps = 10000

    
    # scheduler = RampUpScheduler(None, lr, lr_ramp_down)
    # scheduler = RampUpCosineDecayScheduler(None, lr, 1e-8, steps, lr_ramp_down)
    # scheduler = RampUpExponentialDecayScheduler(None, lr, 1e-8, steps, lr_ramp_down)
    # scheduler = RampUpLogisticDecayScheduler(None, lr, 1e-8, steps, lr_ramp_down)
    scheduler = RampUpSteps(None, {1000:1e-3, 5000:5e-4, 10000:1e-4},100)
    # scheduler = RampUpCosineAnnealingScheduler(None, lr, 1e-8, lr_ramp_down, 1000, 2, 4)

    lrs = []
    for i in range(steps):
        lr = scheduler.get_last_lr()
        scheduler._current_step+=1
        lrs.append(lr)
        
    print(lrs[0], lrs[-1])
    matplotlib.pyplot.plot(lrs)
    matplotlib.pyplot.show()