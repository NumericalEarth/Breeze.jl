using Breeze: Breeze
using Aqua: Aqua
using ExplicitImports: ExplicitImports
using Test: @testset, @test

# Needed only to trigger loading the extensions, so that `ExplicitImports` can analyse them.
using CloudMicrophysics: CloudMicrophysics
using ClimaComms: ClimaComms
using Dates: Dates
using RRTMGP: RRTMGP

@testset "Aqua" begin
    Aqua.test_all(Breeze)
end

@testset "ExplicitImports" begin
    @testset "Explicit Imports" begin
        @test ExplicitImports.check_no_implicit_imports(Breeze) === nothing
    end

    @testset "Import via Owner" begin
        @test ExplicitImports.check_all_explicit_imports_via_owners(Breeze) === nothing
    end

    @testset "Stale Explicit Imports" begin
        @test ExplicitImports.check_no_stale_explicit_imports(Breeze) === nothing
    end

    @testset "Qualified Accesses" begin
        @test ExplicitImports.check_all_qualified_accesses_via_owners(Breeze) === nothing
    end

    @testset "Self Qualified Accesses" begin
        @test ExplicitImports.check_no_self_qualified_accesses(Breeze) === nothing
    end
end
