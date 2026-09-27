# Unit, Numerical, and Integration Test Targets

# Smoke test
add_executable(test_smoke unit/test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE kode_core)
add_test(NAME SmokeTest COMMAND test_smoke)

# Tensor Engine unit test
add_executable(test_tensor unit/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE kode_core)
add_test(NAME TensorUnitTest COMMAND test_tensor)

# Autodiff numerical gradcheck test
add_executable(test_gradcheck numerical/test_gradcheck.cpp)
target_link_libraries(test_gradcheck PRIVATE kode_core)
add_test(NAME GradCheckTest COMMAND test_gradcheck)

# Neural Network framework unit test
add_executable(test_nn unit/test_nn.cpp)
target_link_libraries(test_nn PRIVATE kode_core)
add_test(NAME NNUnitTest COMMAND test_nn)

# Text Processing & Conditioning unit test
add_executable(test_text unit/test_text.cpp)
target_link_libraries(test_text PRIVATE kode_core)
add_test(NAME TextUnitTest COMMAND test_text)

# Image & Data Subsystem unit test
add_executable(test_image_data unit/test_image_data.cpp)
target_link_libraries(test_image_data PRIVATE kode_core)
add_test(NAME ImageDataUnitTest COMMAND test_image_data)

# Model Assembly unit test (Phase 7)
add_executable(test_model unit/test_model.cpp)
target_link_libraries(test_model PRIVATE kode_core)
add_test(NAME ModelUnitTest COMMAND test_model)

# Training Engine unit test (Phase 8)
add_executable(test_training unit/test_training.cpp)
target_link_libraries(test_training PRIVATE kode_core)
add_test(NAME TrainingUnitTest COMMAND test_training)

# Inference & Sampling unit test (Phase 9)
add_executable(test_inference unit/test_inference.cpp)
target_link_libraries(test_inference PRIVATE kode_core)
add_test(NAME InferenceUnitTest COMMAND test_inference)

# Evaluation Subsystem unit test (Phase 10)
add_executable(test_evaluation unit/test_evaluation.cpp)
target_link_libraries(test_evaluation PRIVATE kode_core)
add_test(NAME EvaluationUnitTest COMMAND test_evaluation)

# Optimization Subsystem unit test (Phase 11)
add_executable(test_optimization unit/test_optimization.cpp)
target_link_libraries(test_optimization PRIVATE kode_core)
add_test(NAME OptimizationUnitTest COMMAND test_optimization)

# Web Interface & REST API unit test (Phase 12)
add_executable(test_web unit/test_web.cpp)
target_link_libraries(test_web PRIVATE kode_core)
add_test(NAME WebUnitTest COMMAND test_web)

# Integration & Regression Test Targets (Phase 13)
# E2E Pipeline Integration Test
add_executable(test_pipeline_e2e integration/test_pipeline_e2e.cpp)
target_link_libraries(test_pipeline_e2e PRIVATE kode_core)
add_test(NAME PipelineE2EIntegrationTest COMMAND test_pipeline_e2e)

# Checkpoint Roundtrip & Robustness Integration Test
add_executable(test_checkpoint_roundtrip integration/test_checkpoint_roundtrip.cpp)
target_link_libraries(test_checkpoint_roundtrip PRIVATE kode_core)
add_test(NAME CheckpointRoundtripIntegrationTest COMMAND test_checkpoint_roundtrip)

# Concurrency & Stress Integration Test
add_executable(test_concurrency_stress integration/test_concurrency_stress.cpp)
target_link_libraries(test_concurrency_stress PRIVATE kode_core)
add_test(NAME ConcurrencyStressIntegrationTest COMMAND test_concurrency_stress)

# Numerical Consistency Integration Test
add_executable(test_numerical_consistency integration/test_numerical_consistency.cpp)
target_link_libraries(test_numerical_consistency PRIVATE kode_core)
add_test(NAME NumericalConsistencyIntegrationTest COMMAND test_numerical_consistency)

# Extended Numerical GradCheck Test
add_executable(test_extended_gradcheck numerical/test_extended_gradcheck.cpp)
target_link_libraries(test_extended_gradcheck PRIVATE kode_core)
add_test(NAME ExtendedGradCheckTest COMMAND test_extended_gradcheck)

# Regression & Determinism Test
add_executable(test_regression_determinism numerical/test_regression_determinism.cpp)
target_link_libraries(test_regression_determinism PRIVATE kode_core)
add_test(NAME RegressionDeterminismTest COMMAND test_regression_determinism)
