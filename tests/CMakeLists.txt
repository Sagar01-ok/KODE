# Unit, Numerical, and Integration Test Targets

# Smoke test
add_executable(test_smoke unit/test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE kode_core)
add_test(NAME SmokeTest COMMAND test_smoke)
