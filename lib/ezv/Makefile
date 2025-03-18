### config section ###

ENABLE_SCOTCH	:= 1

CFLAGS		:= -O3 -march=native

######################

ARCH		:= $(shell uname -s | tr a-z A-Z)

PSOURCES	:= $(wildcard test/*.c)
POBJECTS	:= $(PSOURCES:test/%.c=obj/%.o)
PDEPENDS	:= $(PSOURCES:test/%.c=deps/%.d)
PROGRAMS	:= $(PSOURCES:test/%.c=bin/%)

LIB			:= lib/libezv.a

SOURCES		:= $(wildcard src/*.c)
OBJECTS		:= $(SOURCES:src/%.c=obj/%.o)
DEPENDS		:= $(SOURCES:src/%.c=deps/%.d)

MAKEFILES	:= Makefile

CFLAGS		+= -Wall -Wno-unused-function -DGL_SILENCE_DEPRECATION
CFLAGS		+= -I./include
INTERNAL	:= -I./internal
LDLIBS		+= -lm

ifeq ($(ENABLE_SCOTCH), 1)
CFLAGS		+= -DENABLE_SCOTCH
ifeq ($(ARCH),LINUX)
CFLAGS		+= -I/usr/include/scotch
endif
LDLIBS		+= -lscotch -lscotcherr
endif

PACKAGES	:= sdl2 cglm

# cglm is used as a header-only package
L_PACKAGES	:= $(filter-out cglm, $(PACKAGES))

ifeq ($(ARCH),DARWIN)
LDLIBS		+= -framework OpenGL
else
CFLAGS		+= -rdynamic
LDFLAGS		+= -export-dynamic
LDLIBS		+= -lGL -ldl
endif

# Query CFLAGS and LDLIBS for all packages
PKG_CHECK	:= $(shell if pkg-config --print-errors --exists $(PACKAGES); then echo 0; else echo 1; fi)
ifeq ($(PKG_CHECK), 1)
$(error Installation problem: missing package)
endif
CFLAGS		+= $(shell pkg-config --cflags $(PACKAGES))
LDFLAGS		+= $(shell pkg-config --libs-only-L $(L_PACKAGES))
LDLIBS		+= $(shell pkg-config --libs-only-l $(L_PACKAGES))


# Must be the first rule
.PHONY: default lib exec
default: lib

lib: $(LIB)

exec: $(PROGRAMS)

# Compile rules

$(OBJECTS) $(POBJECTS): $(MAKEFILES)

$(PROGRAMS): bin/%: obj/%.o $(LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(OBJECTS): obj/%.o: src/%.c
	$(CC) -o $@ $(CFLAGS) $(INTERNAL) -c $<

$(POBJECTS): obj/%.o: test/%.c
	$(CC) -o $@ $(CFLAGS) -c $<

$(LIB): $(OBJECTS)
	ar rcs $@ $^


$(DEPENDS) $(PDEPENDS): $(MAKEFILES)

$(DEPENDS): deps/%.d: src/%.c
	$(CC) -o $@ $(CFLAGS) $(INTERNAL) -MM -MT "deps/$*.d obj/$*.o" $<

$(PDEPENDS): deps/%.d: test/%.c
	$(CC) -o $@ $(CFLAGS) -MM -MT "deps/$*.d obj/$*.o" $<

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPENDS)
ifeq ($(MAKECMDGOALS),exec)
-include $(PDEPENDS)
endif
endif

.PHONY: clean
clean:
	rm -f $(PROGRAMS) obj/*.o deps/*.d lib/*
