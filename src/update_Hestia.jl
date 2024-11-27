function output_matrix( geometry :: AbstractCubicObject, iosetup :: IOSetup, num_sensors)

    Nc = sum(geometry.heatcells)
    C = zeros(num_sensors,Nc)
    b_pos = getboundarypositions( geometry  )

    for bp in b_pos
        ids = unique(iosetup.identifier[bp])
        for i in ids
            idx = findall(x-> x==i, iosetup.identifier[bp])
            boundary_idx = iosetup.indices[bp]
            boundary_char = iosetup.character[bp]
            C[i,boundary_idx[idx]] = boundary_char[idx] / sum(boundary_char[idx])
        end
    end

    return C
end