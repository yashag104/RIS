/*
 * ==========================================================================
 * TORUS TOPOLOGY PATCH FOR NOXIM
 * ==========================================================================
 * 
 * This file contains the instructions and code to add Torus topology
 * support to Noxim. The Torus is a Mesh with wrap-around links connecting
 * edge routers, reducing diameter from 2*(N-1) to N and average hops by ~33%.
 *
 * FILES TO MODIFY:
 *   1. src/GlobalParams.h   — Add TOPOLOGY_TORUS constant
 *   2. src/NoC.h            — Add buildTorus() declaration
 *   3. src/NoC.cpp          — Add buildTorus() implementation + constructor case
 *   4. src/Router.cpp       — Add TORUS routing (shortest-path XY with wrap)
 *   5. src/routingAlgorithms — Add Routing_TORUS_XY files
 *   6. src/ConfigurationManager.cpp — Parse "TORUS" topology string
 *
 * REBUILD:
 *   cd noxim/bin && make clean && make
 *
 * ==========================================================================
 */

// =====================================================================
// STEP 1: src/GlobalParams.h
// Add after: #define TOPOLOGY_MESH "MESH"
// =====================================================================
//
//   #define TOPOLOGY_TORUS "TORUS"
//

// =====================================================================
// STEP 2: src/NoC.h  
// Add in private section after: void buildMesh();
// =====================================================================
//
//   void buildTorus();
//

// =====================================================================
// STEP 3: src/NoC.h — Constructor
// Add in SC_CTOR after: else if (GlobalParams::topology == TOPOLOGY_OMEGA)
// =====================================================================
//
//   else if (GlobalParams::topology == TOPOLOGY_TORUS)
//       buildTorus();
//

// =====================================================================
// STEP 4: src/NoC.cpp — buildTorus() implementation
// Add at the end of the file (after buildMesh())
// =====================================================================

/*
 * PASTE THIS FUNCTION INTO src/NoC.cpp:
 *
 * buildTorus() is identical to buildMesh() except it adds wrap-around
 * connections for edge routers:
 *   - Row wrap: (x=dim_x-1, y) ↔ (x=0, y) for all y
 *   - Col wrap: (x, y=dim_y-1) ↔ (x, y=0) for all x
 */

/*

void NoC::buildTorus()
{
    // Step 1: Build the standard mesh first
    buildMesh();
    
    // Step 2: Add wrap-around connections (Torus links)
    int dimx = GlobalParams::mesh_dim_x;
    int dimy = GlobalParams::mesh_dim_y;
    
    // Row wraps: connect last column to first column (East-West wrap)
    for (int j = 0; j < dimy; j++)
    {
        // East port of (dimx-1, j) connects to West port of (0, j)
        // Signals: req, ack, buffer_full_status, flit, free_slots, nop_data
        
        int right_id = j * dimx + (dimx - 1);  // rightmost node in row j
        int left_id  = j * dimx + 0;            // leftmost node in row j
        
        // The wrap-around uses the same signal types as regular mesh links.
        // We need to create new sc_signal instances for the wrap links.
        // 
        // NOTE: Since Noxim pre-allocates signals in buildMesh() based on
        // the grid, we use the existing signal infrastructure.
        // The simplest approach is to directly connect ports:
        
        // East output of rightmost → West input of leftmost
        t[dimx-1][j]->req_tx[DIRECTION_EAST](req[right_id][0].east);
        t[0][j]->req_rx[DIRECTION_WEST](req[right_id][0].east);
        
        t[dimx-1][j]->flit_tx[DIRECTION_EAST](flit[right_id][0].east);
        t[0][j]->flit_rx[DIRECTION_WEST](flit[right_id][0].east);
        
        t[dimx-1][j]->ack_rx[DIRECTION_EAST](ack[right_id][0].east);
        t[0][j]->ack_tx[DIRECTION_WEST](ack[right_id][0].east);
        
        t[dimx-1][j]->buffer_full_status_rx[DIRECTION_EAST](buffer_full_status[right_id][0].east);
        t[0][j]->buffer_full_status_tx[DIRECTION_WEST](buffer_full_status[right_id][0].east);
        
        t[dimx-1][j]->free_slots_rx[DIRECTION_EAST](free_slots[right_id][0].east);
        t[0][j]->free_slots_tx[DIRECTION_WEST](free_slots[right_id][0].east);
        
        t[dimx-1][j]->nop_data_rx[DIRECTION_EAST](nop_data[right_id][0].east);
        t[0][j]->nop_data_tx[DIRECTION_WEST](nop_data[right_id][0].east);
        
        // West output of leftmost → East input of rightmost (reverse direction)
        t[0][j]->req_tx[DIRECTION_WEST](req[left_id][0].west);
        t[dimx-1][j]->req_rx[DIRECTION_EAST](req[left_id][0].west);
        
        t[0][j]->flit_tx[DIRECTION_WEST](flit[left_id][0].west);
        t[dimx-1][j]->flit_rx[DIRECTION_EAST](flit[left_id][0].west);
        
        t[0][j]->ack_rx[DIRECTION_WEST](ack[left_id][0].west);
        t[dimx-1][j]->ack_tx[DIRECTION_EAST](ack[left_id][0].west);
        
        t[0][j]->buffer_full_status_rx[DIRECTION_WEST](buffer_full_status[left_id][0].west);
        t[dimx-1][j]->buffer_full_status_tx[DIRECTION_EAST](buffer_full_status[left_id][0].west);
        
        t[0][j]->free_slots_rx[DIRECTION_WEST](free_slots[left_id][0].west);
        t[dimx-1][j]->free_slots_tx[DIRECTION_EAST](free_slots[left_id][0].west);
        
        t[0][j]->nop_data_rx[DIRECTION_WEST](nop_data[left_id][0].west);
        t[dimx-1][j]->nop_data_tx[DIRECTION_EAST](nop_data[left_id][0].west);
    }
    
    // Column wraps: connect last row to first row (North-South wrap)
    for (int i = 0; i < dimx; i++)
    {
        int bottom_id = (dimy - 1) * dimx + i;  // bottom node in column i
        int top_id    = 0 * dimx + i;            // top node in column i
        
        // South output of bottom → North input of top
        t[i][dimy-1]->req_tx[DIRECTION_SOUTH](req[bottom_id][0].south);
        t[i][0]->req_rx[DIRECTION_NORTH](req[bottom_id][0].south);
        
        t[i][dimy-1]->flit_tx[DIRECTION_SOUTH](flit[bottom_id][0].south);
        t[i][0]->flit_rx[DIRECTION_NORTH](flit[bottom_id][0].south);
        
        t[i][dimy-1]->ack_rx[DIRECTION_SOUTH](ack[bottom_id][0].south);
        t[i][0]->ack_tx[DIRECTION_NORTH](ack[bottom_id][0].south);
        
        t[i][dimy-1]->buffer_full_status_rx[DIRECTION_SOUTH](buffer_full_status[bottom_id][0].south);
        t[i][0]->buffer_full_status_tx[DIRECTION_NORTH](buffer_full_status[bottom_id][0].south);
        
        t[i][dimy-1]->free_slots_rx[DIRECTION_SOUTH](free_slots[bottom_id][0].south);
        t[i][0]->free_slots_tx[DIRECTION_NORTH](free_slots[bottom_id][0].south);
        
        t[i][dimy-1]->nop_data_rx[DIRECTION_SOUTH](nop_data[bottom_id][0].south);
        t[i][0]->nop_data_tx[DIRECTION_NORTH](nop_data[bottom_id][0].south);
        
        // North output of top → South input of bottom (reverse)
        t[i][0]->req_tx[DIRECTION_NORTH](req[top_id][0].north);
        t[i][dimy-1]->req_rx[DIRECTION_SOUTH](req[top_id][0].north);
        
        t[i][0]->flit_tx[DIRECTION_NORTH](flit[top_id][0].north);
        t[i][dimy-1]->flit_rx[DIRECTION_SOUTH](flit[top_id][0].north);
        
        t[i][0]->ack_rx[DIRECTION_NORTH](ack[top_id][0].north);
        t[i][dimy-1]->ack_tx[DIRECTION_SOUTH](ack[top_id][0].north);
        
        t[i][0]->buffer_full_status_rx[DIRECTION_NORTH](buffer_full_status[top_id][0].north);
        t[i][dimy-1]->buffer_full_status_tx[DIRECTION_SOUTH](buffer_full_status[top_id][0].north);
        
        t[i][0]->free_slots_rx[DIRECTION_NORTH](free_slots[top_id][0].north);
        t[i][dimy-1]->free_slots_tx[DIRECTION_SOUTH](free_slots[top_id][0].north);
        
        t[i][0]->nop_data_rx[DIRECTION_NORTH](nop_data[top_id][0].north);
        t[i][dimy-1]->nop_data_tx[DIRECTION_SOUTH](nop_data[top_id][0].north);
    }
}

*/

// =====================================================================
// STEP 5: src/routingAlgorithms/Routing_TORUS_XY.h (NEW FILE)
// Shortest-path XY routing for Torus (wraps around if shorter)
// =====================================================================

/*

#ifndef __ROUTING_TORUS_XY_H__
#define __ROUTING_TORUS_XY_H__

#include "RoutingAlgorithm.h"
#include "../Router.h"

class Routing_TORUS_XY : public RoutingAlgorithm
{
  public:
    vector<int> route(Router *router, const RouteData &route_data);
    static Routing_TORUS_XY *getInstance();

  private:
    Routing_TORUS_XY(){};
    static Routing_TORUS_XY *instance;
};

#endif

*/

// =====================================================================
// STEP 6: src/routingAlgorithms/Routing_TORUS_XY.cpp (NEW FILE)
// =====================================================================

/*

#include "Routing_TORUS_XY.h"
#include "../GlobalParams.h"

Routing_TORUS_XY *Routing_TORUS_XY::instance = 0;

Routing_TORUS_XY *Routing_TORUS_XY::getInstance()
{
    if (instance == 0)
        instance = new Routing_TORUS_XY();
    return instance;
}

vector<int> Routing_TORUS_XY::route(Router *router, const RouteData &route_data)
{
    // Torus-aware XY routing: choose shortest path (direct or wrap-around)
    Coord current = route_data.current_id;
    Coord dest    = route_data.dest_id;
    vector<int> directions;

    int dimx = GlobalParams::mesh_dim_x;
    int dimy = GlobalParams::mesh_dim_y;

    int dx = dest.x - current.x;
    int dy = dest.y - current.y;

    // X dimension: choose shortest path
    if (dx != 0) {
        // Direct distance vs wrap-around distance
        int direct_dist = abs(dx);
        int wrap_dist   = dimx - direct_dist;
        
        if (direct_dist <= wrap_dist) {
            // Go directly
            directions.push_back(dx > 0 ? DIRECTION_EAST : DIRECTION_WEST);
        } else {
            // Go via wrap-around (opposite direction)
            directions.push_back(dx > 0 ? DIRECTION_WEST : DIRECTION_EAST);
        }
    } else if (dy != 0) {
        // Y dimension (only if X is aligned — XY routing)
        int direct_dist = abs(dy);
        int wrap_dist   = dimy - direct_dist;
        
        if (direct_dist <= wrap_dist) {
            directions.push_back(dy > 0 ? DIRECTION_SOUTH : DIRECTION_NORTH);
        } else {
            directions.push_back(dy > 0 ? DIRECTION_NORTH : DIRECTION_SOUTH);
        }
    }

    return directions;
}

*/

// =====================================================================
// STEP 7: src/routingAlgorithms/RoutingAlgorithms.h
// Add include:
//   #include "Routing_TORUS_XY.h"
//
// STEP 8: src/routingAlgorithms/RoutingAlgorithms.cpp
// Add in getRoutingAlgorithmInstance():
//   if (GlobalParams::routing_algorithm == "TORUS_XY")
//       return Routing_TORUS_XY::getInstance();
//
// STEP 9: src/ConfigurationManager.cpp
// Add TORUS to topology parsing:
//   else if (topology == TOPOLOGY_TORUS)
//       GlobalParams::topology = TOPOLOGY_TORUS;
// =====================================================================

// =====================================================================
// STEP 10: bin/Makefile (or CMakeLists.txt)
// Add Routing_TORUS_XY.cpp to the build sources
// =====================================================================
