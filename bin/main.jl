ENV["DRIVER_SCRIPT"] = @__FILE__

# actual distributed code

## The following are optional
# Setting RESULTS_FILE_TO_UPLOAD will upload the file to cloud storage - can be downloaded from UI
# Setting OUTPUTS will show the outputs in the UI directly.
ENV["RESULTS_FILE_TO_UPLOAD"] = "<<< RESULTS FILE IF ANY >>>"
ENV["OUTPUTS"] = JSON.json(Dict(:a => "something", :b => "something else"))


function main()


end

main()

for i in workers()
    rmprocs(i)
end
