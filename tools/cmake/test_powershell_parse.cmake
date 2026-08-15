# Syntax check for every PowerShell harness in benchmarks/ (#442).
#
# These scripts run on exactly one machine each, by hand, minutes into a session
# that has already spent ten of them building. A syntax error therefore surfaces
# at the worst possible moment and on the one machine nobody else has -- which is
# not hypothetical: PowerShell defects reached main twice (#437/#438) because
# nothing in CI ever loaded these files.
#
# Parsing is not running, and does not pretend to be: it catches syntax and
# missing-brace errors, not logic. That is precisely the class that costs a whole
# session, and it costs nothing to check.
#
# Expects: INTERP, SCRIPTS (a ;-list of .ps1 paths)

foreach (script ${SCRIPTS})
    if(NOT EXISTS "${script}")
        message(FATAL_ERROR "no such PowerShell script: ${script}")
    endif()
    # ParseFile reports every error it finds, not just the first, so one run
    # names the whole set.
    execute_process(
        COMMAND ${INTERP} -NoProfile -NonInteractive -Command
                "$errors = $null; $null = [System.Management.Automation.Language.Parser]::ParseFile('${script}', [ref]$null, [ref]$errors); if ($errors) { $errors | ForEach-Object { Write-Output $_.ToString() }; exit 1 }; exit 0"
        RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "${script} does not parse:\n${_out}${_err}")
    endif()
endforeach()

message(STATUS "PowerShell harnesses parse cleanly")
