
import sys, os

from shapely import geometry, affinity, ops

def points_and_paths(thing):
    points = [ ]
    paths = [ ]
    point_index = { }
    def get_point(pos):
        if pos not in point_index:
            point_index[pos] = len(points)
            points.append(list(pos))
        return point_index[pos]
    
    def add_ring(ring):
        paths.append([ get_point(item) for item in ring.coords ])
    
    def add(thing):
        if type(thing) is geometry.Polygon:
            add_ring(thing.exterior)
            for item in thing.interiors:
                add_ring(item)
        elif type(thing) is geometry.MultiPolygon:
            for item in thing.geoms:
                add(item)
        else:
            assert False, repr(type(thing))

    add(thing)
    return points, paths


def as_openscad(thing):
    points, paths = points_and_paths(thing)        
    return 'polygon(points=%s,paths=%s)' % (repr(points),repr(paths))


def as_patch(thing, **kwargs):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    points, paths = points_and_paths(thing)
    
    vertices = [ ]
    codes = [ ]
    for item in paths:
        codes.extend([Path.MOVETO]+[Path.LINETO]*(len(item)-1))
        vertices.extend( points[item2] for item2 in item )
    path = Path(vertices, codes)
    return PathPatch(path, **kwargs)
        

def view(thing):
    from matplotlib import pyplot

    pyplot.figure()
    pyplot.gca().add_patch(as_patch(thing))    
    (minx, miny, maxx, maxy) = thing.bounds
    pyplot.xlim(minx-5,maxx+5)
    pyplot.ylim(miny-5,maxy+5)
    pyplot.gca().set_aspect(1)
    #pyplot.show()


def make(prefix, thing):
    # Double size
    thing_big = affinity.scale(thing,2.0,2.0)
    
    (minx, miny, maxx, maxy) = thing_big.bounds
    thing_shifted = affinity.translate(thing_big,
        -(minx+maxx)/2.0,
        -(miny+maxy)/2.0)

    with open(prefix+'.scad','wb') as f:
        f.write(
             'union() { ' +             
             'linear_extrude(height=4) '+
             as_openscad(thing_shifted.buffer(-0.6))+';'
             'translate([0,0,0.5]) ' +
             'linear_extrude(height=3.0) '+
             as_openscad(thing_shifted.buffer(-0.1))+';' # Allow 0.2mm between pieces
             '}'
             )

    (minx, miny, maxx, maxy) = thing.bounds
    thing_shifted = affinity.translate(thing,
         5-minx,
         5-miny)
    
    with open(prefix+'-2D.scad','wb') as f: 
        f.write(
             as_openscad(thing_shifted) + ';'
             )
    
    print 'Build', prefix
    assert 0 == os.system(
        'openscad -o %s.stl %s.scad' % (prefix,prefix)
        )
    assert 0 == os.system(
        'openscad -o %s-2D.dxf %s-2D.scad' % (prefix,prefix)
        )
    

directions = [(1,0),(0,1),(-1,0),(0,-1)]

def points(spec):
    x = 0
    y = 0
    d = 0
    result = [ ]
    for char in spec:
        if char == 'n':
            if not result or result[-1] != (x,y):
                result.append((x,y))
        elif char == 'l':
            d = (d+1)%4
        elif char == 'r':
            d = (d-1)%4
        elif char == 'f':
            x += directions[d][0]
            y += directions[d][1]
        else:
            assert char == ' '
    return result

def mirror(spec):
    mapper = {'l':'r','r':'l'}
    return ''.join(mapper.get(char,char) for char in spec)

def bigger(spec, scale=2):
    return spec.replace('f','f'*scale)



#nub_left = 'nff n lffnlfn rffrffn ffrffn rfnlffn l ffn'
nub_left = 'nff n lffnlfn rffrffn ffrffn rfnlffn l rfnlfnflfnr'
nublet_left = 'n lfnlfn rffrffn ffrffn rfnlfn l'
bump_left = 'n lfrfn frfln'
nub_right = mirror(nub_left[::-1]) #mirror(nub_left)
nublet_right = mirror(nublet_left)
bump_right = mirror(bump_left)
flat = 'n ffn'
a = 'nff'
b = 'ffnl'


def cell(filled, left,mid,right):
    outline = (
        a +
        #flat + 
        (nub_right if filled else flat*3) + 
        #bump_left +
        b +
        
        a + 
        flat + 
        (nublet_left if right else flat) + 
        (nublet_right if mid else flat) + 
        b +
        
        a +
        #bump_right + 
        (nub_left if mid else flat*3) + 
        #flat + 
        b +
        
        a + 
        (nublet_left if left else flat) +
        (nublet_right if mid else flat) +
        flat +
        b
        )
    shape = geometry.Polygon(points(outline))
    if not filled:
       shape = shape.difference(shape.buffer(-2).buffer(1))
    return shape


big_bang = geometry.Polygon(points(
    a + nub_right + b +
    (a + flat*3 + b) * 3
    ))

big_bang = big_bang.difference(geometry.Polygon(points(
    'ffflfffffffffr' +
    bigger(nublet_right + 'll' + flat, 2)
    )))


if __name__ == '__main__':
    if not os.path.exists('output'):
       os.mkdir('output')
    
    cells = [ ]
    
    rule = int(sys.argv[1])
    
    for i in xrange(8):
        right = i&1
        mid = (i>>1)&1
        left = (i>>2)&1
        filled = (rule>>i)&1
        #if not (right or mid or left or filled): continue
        cells.append(
            affinity.translate(cell(filled, left,mid,right), (i%4-2)*15, (i//4-1)*15)
            )
    
    union_cells = ops.cascaded_union(cells)
    
    make('output/big-bang', big_bang)
    make('output/rule-%d-cells' % rule, union_cells)
    
    #view(big_bang)
    #view(union_cells)
    #from matplotlib import pyplot
    #pyplot.show()
    
    
