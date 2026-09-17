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
