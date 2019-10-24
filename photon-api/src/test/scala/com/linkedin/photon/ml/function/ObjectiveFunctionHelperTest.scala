/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.function

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.function.glm.DistributedGLMLossFunction
import com.linkedin.photon.ml.function.svm.DistributedSmoothedHingeLossFunction
import com.linkedin.photon.ml.optimization.game.FixedEffectOptimizationConfiguration
import com.linkedin.photon.ml.optimization.{OptimizerConfig, OptimizerType}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Unit tests for [[ObjectiveFunctionHelper]].
 */
class ObjectiveFunctionHelperTest {

  import ObjectiveFunctionHelperTest._

  @DataProvider
  def trainingTaskProvider(): Array[Array[Any]] =
    Array(
      Array(TaskType.LOGISTIC_REGRESSION),
      Array(TaskType.LINEAR_REGRESSION),
      Array(TaskType.POISSON_REGRESSION),
      Array(TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM))

  /**
   * Test that the [[ObjectiveFunction]] generated by the factory function returned by the [[ObjectiveFunctionHelper]]
   * is of the appropriate type for the given training task.
   *
   * @param trainingTask The training task
   */
  @Test(dataProvider = "trainingTaskProvider")
  def testBuildFactory(trainingTask: TaskType): Unit = {

    val objectiveFunction = ObjectiveFunctionHelper.buildFactory(
      trainingTask,
      TREE_AGGREGATE_DEPTH)(
      COORDINATE_OPT_CONFIG,
      ENABLE_INCREMENTAL_TRAINING)

    trainingTask match {
      case TaskType.LOGISTIC_REGRESSION | TaskType.LINEAR_REGRESSION | TaskType.POISSON_REGRESSION =>
        assertTrue(
          objectiveFunction.isInstanceOf[(Option[GeneralizedLinearModel], Option[Int]) => DistributedGLMLossFunction])

      case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
        assertTrue(
          objectiveFunction
            .isInstanceOf[(Option[GeneralizedLinearModel], Option[Int]) => DistributedSmoothedHingeLossFunction])
    }
  }
}

object ObjectiveFunctionHelperTest {

  val COORDINATE_OPT_CONFIG = FixedEffectOptimizationConfiguration(OptimizerConfig(OptimizerType.LBFGS, 1, 2e-2))
  val ENABLE_INCREMENTAL_TRAINING = false
  val MAXIMUM_ITERATIONS = 1
  val TOLERANCE = 2e-2
  val TREE_AGGREGATE_DEPTH = 3
}
