"""Opt-in elapsed-clock trajectory; original absolute-clock default is intact."""
from .trajectory import TrajectoryModel


class RelativeClockTrajectoryModel(TrajectoryModel):
    """Remove arbitrary file origin, retaining observed elapsed time.

    The external four-loss function still uses original timestamps for real
    future deltas. Parameter names/shapes are identical to TrajectoryModel.
    This repairs origin dependence, not spatial supervision.
    """

    def forward(self, teacher, valid, timestamps, *, detach_teacher=True):
        clock=timestamps.masked_fill(~valid,0)
        elapsed=(clock-clock[:,:1]).masked_fill(~valid,0)
        return super().forward(teacher,valid,elapsed,detach_teacher=detach_teacher)


def trajectory_model_for_clock(width,policy="absolute",**kwargs):
    if policy=="absolute":return TrajectoryModel(width,**kwargs)
    if policy=="relative_prefix":return RelativeClockTrajectoryModel(width,**kwargs)
    raise ValueError(f"Unknown clock policy: {policy}")
