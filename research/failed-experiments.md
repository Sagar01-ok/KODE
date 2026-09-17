# KODE Failed Experiments & Negative Results Log

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Guiding Principle:** Every failure is empirical data. Never hide or fictionalize negative outcomes.

---

## Log Entries

### EXP-FAIL-000: Default PATH Compiler Toolchain Invocation
* **Date:** 2026-09-17
* **Hypothesis:** `cl.exe`, `gcc`, or `ninja` would be accessible directly in standard Windows environment `PATH`.
* **Configuration:** Invocation of `gcc --version`, `ninja --version`, `cl.exe` from PowerShell without environment initialization.
* **Result:** Exit code 1 / `CommandNotFoundException` for `gcc`, `ninja`, and `cl.exe`.
* **Investigation & Root Cause:** Modern Windows systems host Visual Studio installations within dedicated directories (`C:\Program Files\Microsoft Visual Studio\2022\Community`) whose compiler executables and Ninja binaries are not placed in the global user PATH by default to prevent DLL/ABI collisions.
* **What Was Learned:** Build scripts, tests, and CMake must target the MSVC toolchain either via CMake's native Visual Studio generator (`-G "Visual Studio 17 2022" -A x64`) or by sourcing `vcvarsall.bat x64`.
* **Resolution:** CMake 4.1.1 is available in PATH and successfully discovers the MSVC 19.44 toolset and MSVC bundled Ninja.

---

### EXP-FAIL-001: CMake generator expression conflict between `/O2` and `/RTC1`
* **Date:** 2026-09-17
* **Hypothesis:** Specifying `$<$<CONFIG:Release>:/O2 ...>` and `$<$<CONFIG:Debug>:/RTC1>` via `add_compile_options` would cleanly map configuration flags in MSVC.
* **Configuration:** Root `CMakeLists.txt` using MSVC generator with Visual Studio 17 2022.
* **Result:** Compilation failed with `cl : command line error D8016: '/O2' and '/RTC1' command-line options are incompatible [E:\KODE\build\kode_core.vcxproj]`.
* **Investigation & Root Cause:** CMake's default multi-configuration Visual Studio generator already inserts `/RTC1` in its internal Debug defaults, causing conflicting flag propagation during project generation.
* **What Was Learned:** In CMake with MSVC, release optimization flags (`/O2 /Oi /Ot /Gy /fp:precise`) must be appended directly to `CMAKE_CXX_FLAGS_RELEASE` rather than mixed via generator expressions inside `add_compile_options`.
* **Resolution:** Replaced the generator expressions with `set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /Ot /Gy /fp:precise")`. Compilation succeeded cleanly.
