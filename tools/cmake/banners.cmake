########################################################################################
# banners.cmake
#
# ASCII art banner for MTL5

macro(print_header)
    message("")
    message("  __  __ _____ _     ____  ")
    message(" |  \\/  |_   _| |   | ___| ")
    message(" | |\\/| | | | | |   |___ \\ ")
    message(" | |  | | | | | |___ ___) |")
    message(" |_|  |_| |_| |_____|____/ ")
    message("")
    message(" Matrix Template Library 5 — C++20 header-only linear algebra")
    message("")
endmacro()

macro(print_footer)
    print_header()
endmacro()
