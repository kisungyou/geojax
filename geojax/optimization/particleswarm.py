"""Derivative-free particle swarm optimizer on manifolds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
import math
import time

import jax
import jax.numpy as jnp

from .minimize import (
    Array,
    InfoEntry,
    StatsFn,
    StopFn,
    as_float,
    cost_value,
    make_info,
    require,
    retract,
    stopping_reason,
    transport,
    tree_lincomb,
    tree_zeros_like,
)


@dataclass(frozen=True)
class ParticleSwarm:
    requires_gradient: bool = False
    swarm_size: int = 30
    inertia: float = 0.5
    cognitive: float = 1.5
    social: float = 1.5
    initial_velocity_scale: float = 0.1
    maxiter: int = 200
    maxtime: float = math.inf
    minstepsize: float = 0.0
    tolgradnorm: float = -math.inf
    verbosity: int = 2
    statsfun: Optional[StatsFn] = None
    stopfun: Optional[StopFn] = None

    def solve(self, problem: Any) -> tuple[Array, float, List[InfoEntry]]:
        M = require(problem, "M")
        x0 = require(problem, "x0")
        n_particles = max(int(self.swarm_size), 2)
        start_time = time.perf_counter()

        particles = [x0]
        for _ in range(n_particles - 1):
            particles.append(M.random_point(problem.split_key()))
        velocities = [
            M.random_tangent(problem.split_key(), x, scale=self.initial_velocity_scale)
            if hasattr(M, "random_tangent")
            else tree_zeros_like(x)
            for x in particles
        ]
        costs = [as_float(cost_value(problem, x)) for x in particles]
        pbest = list(particles)
        pbest_costs = list(costs)
        gidx = int(jnp.argmin(jnp.asarray(pbest_costs)))
        gbest = pbest[gidx]
        gbest_cost = pbest_costs[gidx]

        info: List[InfoEntry] = [
            make_info(
                iter=0,
                cost=gbest_cost,
                gradnorm=math.nan,
                stepsize=math.nan,
                start_time=start_time,
                linesearch=None,
                problem=problem,
                x=gbest,
                solver=self,
            )
        ]
        if self.verbosity >= 2:
            print(" iter\t        best cost\t mean step")
        while True:
            if self.verbosity >= 2:
                print(f"{info[-1].iter:5d}\t{info[-1].cost:+.16e}\t{info[-1].stepsize:.8e}")
            reason = stopping_reason(problem, gbest, info, self)
            if reason:
                info[-1] = InfoEntry(**{**info[-1].__dict__, "reason": reason})
                if self.verbosity >= 1:
                    print(reason)
                break

            new_particles = []
            new_velocities = []
            stepnorms = []
            for x, v, xb, _fb in zip(particles, velocities, pbest, pbest_costs):
                key = problem.split_key()
                r1, r2 = jax.random.uniform(key, shape=(2,))
                to_pbest = M.log(x, xb) if hasattr(M, "log") else tree_zeros_like(x)
                to_gbest = M.log(x, gbest) if hasattr(M, "log") else tree_zeros_like(x)
                if hasattr(M, "lincomb"):
                    v_new = M.lincomb(
                        x,
                        self.inertia,
                        v,
                        self.cognitive * r1,
                        to_pbest,
                        self.social * r2,
                        to_gbest,
                    )
                else:
                    v_new = tree_lincomb(
                        self.inertia,
                        v,
                        self.cognitive * r1,
                        to_pbest,
                        self.social * r2,
                        to_gbest,
                    )
                x_new = retract(M, x, v_new, 1.0)
                v_at_new = transport(M, x, x_new, v_new)
                new_particles.append(x_new)
                new_velocities.append(v_at_new)
                stepnorms.append(as_float(M.norm(x, v_new)) if hasattr(M, "norm") else math.nan)
            particles = new_particles
            velocities = new_velocities
            costs = [as_float(cost_value(problem, x)) for x in particles]
            for i, (x, f) in enumerate(zip(particles, costs)):
                if f < pbest_costs[i]:
                    pbest[i] = x
                    pbest_costs[i] = f
            gidx = int(jnp.argmin(jnp.asarray(pbest_costs)))
            if pbest_costs[gidx] < gbest_cost:
                gbest = pbest[gidx]
                gbest_cost = pbest_costs[gidx]
            mean_step = float(jnp.nanmean(jnp.asarray(stepnorms)))
            info.append(
                make_info(
                    iter=info[-1].iter + 1,
                    cost=gbest_cost,
                    gradnorm=math.nan,
                    stepsize=mean_step,
                    start_time=start_time,
                    linesearch=None,
                    problem=problem,
                    x=gbest,
                    solver=self,
                )
            )
        if self.verbosity >= 1:
            print(f"Total time is {info[-1].time:.6f} [s]")
        return gbest, info[-1].cost, info


__all__ = ["ParticleSwarm"]
