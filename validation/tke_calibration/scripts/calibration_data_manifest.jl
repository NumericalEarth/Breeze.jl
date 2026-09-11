# The GCM file is mutable and includes reconstructed solar forcing. Pin its contents, plus every
# reduced LES member, independently of the source-code protocol version.
using SHA: sha256

file_sha256(path) = open(io -> bytes2hex(sha256(io)), path)

function calibration_data_manifest(member_ids)
    gcm = BreezeCalibration.default_gcm_columns
    return (; gcm_sha256 = file_sha256(gcm),
              les = [(; site, month, sha256 = file_sha256(BreezeCalibration.member_path(site, month)))
                     for (site, month) in member_ids])
end

function validate_data_manifest(saved)
    configuration = saved["run_configuration"]
    if haskey(configuration, :data_manifest)
        current = calibration_data_manifest(saved["members"])
        isequal(configuration.data_manifest, current) || error("Calibration forcing data changed since this checkpoint")
    else
        @warn "Legacy checkpoint has no forcing-data hashes; direct reproduction is still required"
    end
    return nothing
end
