/*
 * Copyright 2020 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.optimization

import breeze.linalg._
import breeze.optimize.NaNHistory
import breeze.math.MutableInnerProductModule

case class ApproximateHessian[T](
  m: Int = 3,
  memStep: IndexedSeq[T] = IndexedSeq.empty,
  memGradDelta: IndexedSeq[T] = IndexedSeq.empty)(implicit space: MutableInnerProductModule[T, Double]) {

  import space._

  def updated(step: T, gradDelta: T): ApproximateHessian[T] = {
    val memStep = (step +: this.memStep).take(m) // add the step in the beginning
    val memGradDelta = (gradDelta +: this.memGradDelta).take(m)

    ApproximateHessian(m, memStep, memGradDelta)
  }

  def historyLength: Int = memStep.length

  def *(grad: T) = {
    val diag = if (historyLength > 0) {
      val prevStep = memStep.head
      val prevGradStep = memGradDelta.head
      val sy = prevStep.dot(prevGradStep)
      val ss = prevStep.dot(prevStep)
      if (sy < 0 || sy.isNaN) throw new NaNHistory // this is used to get diagonal
      sy / ss
    } else {
      1.0
    }

    val dir = space.copy(grad)
    val as = new Array[Double](m)
    val rho = new Array[Double](m)

    for (i <- 0 until historyLength) {
      rho(i) = memStep(i).dot(memGradDelta(i))    // compute rho_i
      as(i) = (memGradDelta(i).dot(dir)) / rho(i)   // alpha = s_i^T r_i / rho_i
      if (as(i).isNaN) {
        throw new NaNHistory
      }
      axpy(-as(i), memStep(i), dir)  // compute -alpha^T * y_i + fir
    }

    dir *= diag

    for (i <- (historyLength - 1) to 0 by (-1)) {
      val beta = (memStep(i).dot(dir)) / rho(i)
      axpy(as(i) - beta, memGradDelta(i), dir)
    }
    dir
  }
}