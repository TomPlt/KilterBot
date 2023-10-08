SELECT 
    c.uuid, c.layout_id, c.setter_id, c.setter_username, c.name, 
    c.frames, c.edge_bottom, c.edge_top, c.edge_left, c.edge_right,
    cs.angle, cs.display_difficulty, cs.benchmark_difficulty,
    cs.ascensionist_count, cs.display_difficulty, cs.quality_average,
    GROUP_CONCAT(bl.link || ',' || bl.foreign_username, ';') AS link_and_username,
    GROUP_CONCAT(CAST(bl.angle AS TEXT), ';') AS beta_angles
FROM 
    climbs c
JOIN 
    climb_stats cs ON c.uuid = cs.climb_uuid
LEFT JOIN 
    beta_links bl ON c.uuid = bl.climb_uuid
JOIN 
    climb_cache_fields cf ON c.uuid = cf.climb_uuid
GROUP BY
    c.uuid, c.layout_id, c.setter_id, c.setter_username, c.name, 
    c.frames, c.edge_bottom, c.edge_top, c.edge_left, c.edge_right,
    cs.angle, cs.display_difficulty, cs.benchmark_difficulty, 
    cf.ascensionist_count, cf.display_difficulty, cf.quality_average
ORDER BY
    cf.ascensionist_count DESC;
