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



