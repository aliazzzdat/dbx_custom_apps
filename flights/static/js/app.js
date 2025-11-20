let flightsTable, passengersTable, airportsTable, airlinesTable, aircraftTable;
let airports = [], airlines = [], flights = [];

// Toast notification function
function showToast(message, type = 'success') {
    const toastId = 'toast-' + Date.now();
    const iconMap = {
        'success': '✓',
        'error': '✗',
        'info': 'ℹ'
    };
    const titleMap = {
        'success': 'Success',
        'error': 'Error',
        'info': 'Information'
    };
    
    const toastHTML = `
        <div id="${toastId}" class="toast custom-toast toast-${type}" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header">
                <strong class="me-auto">${iconMap[type]} ${titleMap[type]}</strong>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    $('#toastContainer').append(toastHTML);
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, {
        autohide: true,
        delay: 3000
    });
    toast.show();
    
    // Remove from DOM after hidden
    toastElement.addEventListener('hidden.bs.toast', function () {
        toastElement.remove();
    });
}

$(document).ready(function() {
    loadDashboard();
    loadReferenceData();
    
    $('#dashboard-tab').on('shown.bs.tab', function() {
        refreshDashboard();
    });
    
    $('#flights-tab').on('shown.bs.tab', function() {
        if (!flightsTable) {
            initFlightsTable();
        } else {
            refreshFlights();
        }
    });
    
    $('#passengers-tab').on('shown.bs.tab', function() {
        if (!passengersTable) {
            initPassengersTable();
        } else {
            refreshPassengers();
        }
    });
    
    $('#airports-tab').on('shown.bs.tab', function() {
        if (!airportsTable) {
            initAirportsTable();
        } else {
            refreshAirports();
        }
    });
    
    $('#airlines-tab').on('shown.bs.tab', function() {
        if (!airlinesTable) {
            initAirlinesTable();
        } else {
            refreshAirlines();
        }
    });
    
    $('#aircraft-tab').on('shown.bs.tab', function() {
        if (!aircraftTable) {
            initAircraftTable();
        } else {
            refreshAircraft();
        }
    });
});

function loadReferenceData() {
    $.get('/api/airports', function(data) {
        airports = data;
        populateAirportDropdowns();
    });
    
    $.get('/api/airlines', function(data) {
        airlines = data;
        populateAirlineDropdowns();
    });
}

function populateAirportDropdowns() {
    const originSelect = $('#originAirport');
    const destSelect = $('#destinationAirport');
    originSelect.empty();
    destSelect.empty();
    
    airports.forEach(airport => {
        const option = `<option value="${airport.airport_code}">
            ${airport.airport_code} - ${airport.airport_name} (${airport.city})
        </option>`;
        originSelect.append(option);
        destSelect.append(option);
    });
}

function populateAirlineDropdowns() {
    const airlineSelect = $('#airlineCode');
    airlineSelect.empty();
    
    airlines.forEach(airline => {
        const option = `<option value="${airline.airline_code}">
            ${airline.airline_code} - ${airline.airline_name}
        </option>`;
        airlineSelect.append(option);
    });
}

function loadFlightsForDropdown() {
    $.get('/api/flights', function(data) {
        flights = data;
        const flightSelect = $('#passengerFlightId');
        flightSelect.empty();
        
        flights.forEach(flight => {
            const option = `<option value="${flight.flight_id}">
                ${flight.flight_number} - ${flight.origin} to ${flight.destination}
            </option>`;
            flightSelect.append(option);
        });
    });
}

function loadDashboard() {
    $.get('/api/kpis', function(data) {
        $('#totalFlights').text(data.total_flights);
        $('#totalPassengers').text(data.total_passengers);
        $('#activeFlights').text(data.active_flights);
        $('#totalAirports').text(data.total_airports);
        $('#totalRevenue').text('$' + (data.total_revenue/1000).toFixed(1) + 'K');
        $('#avgTicketPrice').text('$' + data.avg_ticket_price.toFixed(0));
        
        createAirportsChart(data.top_airports);
        createAirlinesChart(data.top_airlines);
        createStatusChart(data.flight_status);
        createClassChart(data.passenger_class);
        createRevenueByClassChart(data.revenue_by_class);
        createRevenueByAirlineChart(data.revenue_by_airline);
        createTopRoutesChart(data.top_routes);
        createDelayByAirportChart(data.delay_by_airport);
        createDelayByAirlineChart(data.delay_by_airline);
    });
}

// Refresh functions
function refreshDashboard() {
    // Clear existing charts
    $('#dashboard canvas').each(function() {
        const chart = Chart.getChart(this);
        if (chart) {
            chart.destroy();
        }
    });
    loadDashboard();
    showToast('Dashboard refreshed', 'info');
}

function refreshFlights() {
    if (flightsTable) {
        flightsTable.ajax.reload(null, false); // false = stay on current page
        showToast('Flights refreshed', 'info');
    }
}

function refreshPassengers() {
    if (passengersTable) {
        passengersTable.ajax.reload(null, false);
        showToast('Passengers refreshed', 'info');
    }
}

function refreshAirports() {
    if (airportsTable) {
        airportsTable.ajax.reload(null, false);
        loadReferenceData(); // Also refresh dropdowns
        showToast('Airports refreshed', 'info');
    }
}

function refreshAirlines() {
    if (airlinesTable) {
        airlinesTable.ajax.reload(null, false);
        loadReferenceData(); // Also refresh dropdowns
        showToast('Airlines refreshed', 'info');
    }
}

function refreshAircraft() {
    if (aircraftTable) {
        aircraftTable.ajax.reload(null, false);
        showToast('Aircraft refreshed', 'info');
    }
}

function createAirportsChart(data) {
    const ctx = document.getElementById('airportsChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.airport),
            datasets: [{
                label: 'Number of Flights',
                data: data.map(d => d.count),
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

function createAirlinesChart(data) {
    const ctx = document.getElementById('airlinesChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.airline),
            datasets: [{
                label: 'Number of Passengers',
                data: data.map(d => d.count),
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

function createStatusChart(data) {
    const ctx = document.getElementById('statusChart').getContext('2d');
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: data.map(d => d.status),
            datasets: [{
                data: data.map(d => d.count),
                backgroundColor: [
                    'rgba(255, 99, 132, 0.5)',
                    'rgba(54, 162, 235, 0.5)',
                    'rgba(255, 206, 86, 0.5)',
                    'rgba(75, 192, 192, 0.5)',
                    'rgba(153, 102, 255, 0.5)',
                    'rgba(255, 159, 64, 0.5)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function createClassChart(data) {
    const ctx = document.getElementById('classChart').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.map(d => d.class),
            datasets: [{
                data: data.map(d => d.count),
                backgroundColor: [
                    'rgba(255, 99, 132, 0.5)',
                    'rgba(54, 162, 235, 0.5)',
                    'rgba(255, 206, 86, 0.5)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function createRevenueByClassChart(data) {
    const ctx = document.getElementById('revenueByClassChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.class),
            datasets: [{
                label: 'Revenue ($)',
                data: data.map(d => d.revenue),
                backgroundColor: [
                    'rgba(255, 206, 86, 0.6)',
                    'rgba(54, 162, 235, 0.6)',
                    'rgba(255, 99, 132, 0.6)'
                ],
                borderColor: [
                    'rgba(255, 206, 86, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 99, 132, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Revenue: $' + context.parsed.y.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

function createRevenueByAirlineChart(data) {
    const ctx = document.getElementById('revenueByAirlineChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.airline),
            datasets: [{
                label: 'Revenue ($)',
                data: data.map(d => d.revenue),
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + (value/1000).toFixed(0) + 'K';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Revenue: $' + context.parsed.x.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

function createTopRoutesChart(data) {
    const ctx = document.getElementById('topRoutesChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.route),
            datasets: [{
                label: 'Revenue ($)',
                data: data.map(d => d.revenue),
                backgroundColor: 'rgba(153, 102, 255, 0.6)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + (value/1000).toFixed(0) + 'K';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const idx = context.dataIndex;
                            const revenue = context.parsed.y;
                            const passengers = data[idx].passengers;
                            return [
                                'Revenue: $' + revenue.toLocaleString(),
                                'Passengers: ' + passengers
                            ];
                        }
                    }
                }
            }
        }
    });
}

function createDelayByAirportChart(data) {
    const ctx = document.getElementById('delayByAirportChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.airport),
            datasets: [{
                label: 'Average Delay (minutes)',
                data: data.map(d => d.avg_delay),
                backgroundColor: 'rgba(255, 159, 64, 0.6)',
                borderColor: 'rgba(255, 159, 64, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return value + ' min';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const idx = context.dataIndex;
                            const delay = context.parsed.x.toFixed(1);
                            const flights = data[idx].flights;
                            return [
                                'Avg Delay: ' + delay + ' min',
                                'Delayed Flights: ' + flights
                            ];
                        }
                    }
                }
            }
        }
    });
}

function createDelayByAirlineChart(data) {
    const ctx = document.getElementById('delayByAirlineChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.airline),
            datasets: [{
                label: 'Average Delay (minutes)',
                data: data.map(d => d.avg_delay),
                backgroundColor: 'rgba(255, 99, 132, 0.6)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return value + ' min';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const idx = context.dataIndex;
                            const delay = context.parsed.y.toFixed(1);
                            const flights = data[idx].flights;
                            return [
                                'Avg Delay: ' + delay + ' min',
                                'Delayed Flights: ' + flights
                            ];
                        }
                    }
                }
            }
        }
    });
}

function initFlightsTable() {
    flightsTable = $('#flightsTable').DataTable({
        ajax: {
            url: '/api/flights',
            dataSrc: ''
        },
                columns: [
                    { data: 'flight_id' },
                    { data: 'flight_number' },
                    { data: 'airline' },
                    { data: 'origin' },
                    { data: 'destination' },
                    { data: 'departure_time' },
                    { data: 'arrival_time' },
                    { data: 'status' },
                    { data: 'aircraft_type' },
                    {
                        data: 'delay_minutes',
                        render: function(data) {
                            if (data == 0) {
                                return '<span class="badge bg-success">On Time</span>';
                            } else if (data < 30) {
                                return '<span class="badge bg-warning">' + data + ' min</span>';
                            } else {
                                return '<span class="badge bg-danger">' + data + ' min</span>';
                            }
                        }
                    },
            {
                data: null,
                render: function(data) {
                    return `
                        <div class="table-actions">
                            <button class="btn btn-sm btn-warning" onclick="editFlight(${data.flight_id})">Edit</button>
                            <button class="btn btn-sm btn-danger" onclick="deleteFlight(${data.flight_id})">Delete</button>
                        </div>
                    `;
                }
            }
        ]
    });
}

function initPassengersTable() {
    passengersTable = $('#passengersTable').DataTable({
        ajax: {
            url: '/api/passengers',
            dataSrc: ''
        },
        columns: [
            { data: 'passport_id' },
            { data: 'first_name' },
            { data: 'last_name' },
            { data: 'email' },
            { data: 'phone' },
            { data: 'sex' },
            { data: 'flight_id' },
            { data: 'seat_number' },
            { data: 'class' },
            { 
                data: 'ticket_price',
                render: function(data) {
                    return '$' + parseFloat(data).toFixed(2);
                }
            },
            { data: 'booking_date' },
            {
                data: null,
                render: function(data) {
                    return `
                        <div class="table-actions">
                            <button class="btn btn-sm btn-warning" onclick="editPassenger('${data.passport_id}')">Edit</button>
                            <button class="btn btn-sm btn-danger" onclick="deletePassenger('${data.passport_id}')">Delete</button>
                        </div>
                    `;
                }
            }
        ]
    });
}

function openFlightForm() {
    $('#flightModalTitle').text('Add Flight');
    $('#flightForm')[0].reset();
    $('#flightId').val('');
}

function editFlight(id) {
    $.get('/api/flights/' + id, function(flight) {
        $('#flightModalTitle').text('Edit Flight');
        $('#flightId').val(flight.flight_id);
        $('#flightNumber').val(flight.flight_number);
        $('#airlineCode').val(flight.airline);
        $('#originAirport').val(flight.origin);
        $('#destinationAirport').val(flight.destination);
        $('#departureTime').val(flight.departure_time.replace(' ', 'T').substring(0, 16));
        $('#arrivalTime').val(flight.arrival_time.replace(' ', 'T').substring(0, 16));
        $('#flightStatus').val(flight.status);
        $('#aircraftType').val(flight.aircraft_type);
        $('#delayMinutes').val(flight.delay_minutes || 0);
        
        $('#flightModal').modal('show');
    });
}

function saveFlight() {
    const flightId = $('#flightId').val();
    const flightData = {
        flight_number: $('#flightNumber').val(),
        airline: $('#airlineCode').val(),
        origin: $('#originAirport').val(),
        destination: $('#destinationAirport').val(),
        departure_time: $('#departureTime').val().replace('T', ' ') + ':00',
        arrival_time: $('#arrivalTime').val().replace('T', ' ') + ':00',
        status: $('#flightStatus').val(),
        aircraft_type: $('#aircraftType').val(),
        delay_minutes: parseInt($('#delayMinutes').val()) || 0
    };
    
    if (flightId) {
        flightData.flight_id = parseInt(flightId);
        $.ajax({
            url: '/api/flights/' + flightId,
            method: 'PUT',
            contentType: 'application/json',
            data: JSON.stringify(flightData),
            success: function() {
                $('#flightModal').modal('hide');
                flightsTable.ajax.reload();
                showToast('Flight updated successfully!', 'success');
            },
            error: function() {
                showToast('Error updating flight', 'error');
            }
        });
    } else {
        $.ajax({
            url: '/api/flights',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(flightData),
            success: function() {
                $('#flightModal').modal('hide');
                flightsTable.ajax.reload();
                showToast('Flight added successfully!', 'success');
            },
            error: function() {
                showToast('Error adding flight', 'error');
            }
        });
    }
}

function deleteFlight(id) {
    if (confirm('Are you sure you want to delete this flight?')) {
        $.ajax({
            url: '/api/flights/' + id,
            method: 'DELETE',
            success: function() {
                flightsTable.ajax.reload();
                showToast('Flight deleted successfully!', 'success');
            },
            error: function() {
                showToast('Error deleting flight', 'error');
            }
        });
    }
}

function openPassengerForm() {
    $('#passengerModalTitle').text('Add Passenger');
    $('#passengerForm')[0].reset();
    $('#passportIdOld').val('');
    $('#passportId').prop('disabled', false);
    loadFlightsForDropdown();
}

function editPassenger(id) {
    loadFlightsForDropdown();
    $.get('/api/passengers/' + id, function(passenger) {
        $('#passengerModalTitle').text('Edit Passenger');
        $('#passportIdOld').val(passenger.passport_id);
        $('#passportId').val(passenger.passport_id).prop('disabled', true);
        $('#firstName').val(passenger.first_name);
        $('#lastName').val(passenger.last_name);
        $('#email').val(passenger.email);
        $('#phone').val(passenger.phone);
        $('#sex').val(passenger.sex);
        $('#passengerFlightId').val(passenger.flight_id);
        $('#seatNumber').val(passenger.seat_number);
        $('#passengerClass').val(passenger.class);
        $('#ticketPrice').val(passenger.ticket_price);
        $('#bookingDate').val(passenger.booking_date.replace(' ', 'T').substring(0, 16));
        
        $('#passengerModal').modal('show');
    });
}

function savePassenger() {
    const oldPassportId = $('#passportIdOld').val();
    const passengerData = {
        passport_id: $('#passportId').val().toUpperCase(),
        first_name: $('#firstName').val(),
        last_name: $('#lastName').val(),
        email: $('#email').val(),
        phone: $('#phone').val(),
        sex: $('#sex').val(),
        flight_id: parseInt($('#passengerFlightId').val()),
        seat_number: $('#seatNumber').val(),
        class: $('#passengerClass').val(),
        ticket_price: parseFloat($('#ticketPrice').val()),
        booking_date: $('#bookingDate').val().replace('T', ' ') + ':00'
    };
    
    if (oldPassportId) {
        $.ajax({
            url: '/api/passengers/' + oldPassportId,
            method: 'PUT',
            contentType: 'application/json',
            data: JSON.stringify(passengerData),
            success: function() {
                $('#passengerModal').modal('hide');
                passengersTable.ajax.reload();
                showToast('Passenger updated successfully!', 'success');
            },
            error: function() {
                showToast('Error updating passenger', 'error');
            }
        });
    } else {
        $.ajax({
            url: '/api/passengers',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(passengerData),
            success: function() {
                $('#passengerModal').modal('hide');
                passengersTable.ajax.reload();
                showToast('Passenger added successfully!', 'success');
            },
            error: function() {
                showToast('Error adding passenger', 'error');
            }
        });
    }
}

function deletePassenger(id) {
    if (confirm('Are you sure you want to delete this passenger?')) {
        $.ajax({
            url: '/api/passengers/' + id,
            method: 'DELETE',
            success: function() {
                passengersTable.ajax.reload();
                showToast('Passenger deleted successfully!', 'success');
            },
            error: function() {
                showToast('Error deleting passenger', 'error');
            }
        });
    }
}

// Airports functions
function initAirportsTable() {
    airportsTable = $('#airportsTable').DataTable({
        ajax: {
            url: '/api/airports',
            dataSrc: ''
        },
        columns: [
            { data: 'airport_code' },
            { data: 'airport_name' },
            { data: 'city' },
            { data: 'country' },
            {
                data: null,
                render: function(data) {
                    return `
                        <div class="table-actions">
                            <button class="btn btn-sm btn-warning" onclick="editAirport('${data.airport_code}')">Edit</button>
                            <button class="btn btn-sm btn-danger" onclick="deleteAirport('${data.airport_code}')">Delete</button>
                        </div>
                    `;
                }
            }
        ]
    });
}

function openAirportForm() {
    $('#airportModalTitle').text('Add Airport');
    $('#airportForm')[0].reset();
    $('#airportCodeOld').val('');
    $('#airportCode').prop('disabled', false);
}

function editAirport(code) {
    $.get('/api/airports/' + code, function(airport) {
        $('#airportModalTitle').text('Edit Airport');
        $('#airportCodeOld').val(airport.airport_code);
        $('#airportCode').val(airport.airport_code).prop('disabled', true);
        $('#airportName').val(airport.airport_name);
        $('#airportCity').val(airport.city);
        $('#airportCountry').val(airport.country);
        
        $('#airportModal').modal('show');
    });
}

function saveAirport() {
    const oldCode = $('#airportCodeOld').val();
    const airportData = {
        airport_code: $('#airportCode').val().toUpperCase(),
        airport_name: $('#airportName').val(),
        city: $('#airportCity').val(),
        country: $('#airportCountry').val()
    };
    
    if (oldCode) {
        $.ajax({
            url: '/api/airports/' + oldCode,
            method: 'PUT',
            contentType: 'application/json',
            data: JSON.stringify(airportData),
            success: function() {
                $('#airportModal').modal('hide');
                airportsTable.ajax.reload();
                loadReferenceData();
                showToast('Airport updated successfully!', 'success');
            },
            error: function() {
                showToast('Error updating airport', 'error');
            }
        });
    } else {
        $.ajax({
            url: '/api/airports',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(airportData),
            success: function() {
                $('#airportModal').modal('hide');
                airportsTable.ajax.reload();
                loadReferenceData();
                showToast('Airport added successfully!', 'success');
            },
            error: function() {
                showToast('Error adding airport', 'error');
            }
        });
    }
}

function deleteAirport(code) {
    if (confirm('Are you sure you want to delete this airport?')) {
        $.ajax({
            url: '/api/airports/' + code,
            method: 'DELETE',
            success: function() {
                airportsTable.ajax.reload();
                loadReferenceData();
                showToast('Airport deleted successfully!', 'success');
            },
            error: function() {
                showToast('Error deleting airport', 'error');
            }
        });
    }
}

// Airlines functions
function initAirlinesTable() {
    airlinesTable = $('#airlinesTable').DataTable({
        ajax: {
            url: '/api/airlines',
            dataSrc: ''
        },
        columns: [
            { data: 'airline_code' },
            { data: 'airline_name' },
            { data: 'country' },
            {
                data: null,
                render: function(data) {
                    return `
                        <div class="table-actions">
                            <button class="btn btn-sm btn-warning" onclick="editAirline('${data.airline_code}')">Edit</button>
                            <button class="btn btn-sm btn-danger" onclick="deleteAirline('${data.airline_code}')">Delete</button>
                        </div>
                    `;
                }
            }
        ]
    });
}

function openAirlineForm() {
    $('#airlineModalTitle').text('Add Airline');
    $('#airlineForm')[0].reset();
    $('#airlineCodeOld').val('');
    $('#airlineCodeInput').prop('disabled', false);
}

function editAirline(code) {
    $.get('/api/airlines/' + code, function(airline) {
        $('#airlineModalTitle').text('Edit Airline');
        $('#airlineCodeOld').val(airline.airline_code);
        $('#airlineCodeInput').val(airline.airline_code).prop('disabled', true);
        $('#airlineName').val(airline.airline_name);
        $('#airlineCountry').val(airline.country);
        
        $('#airlineModal').modal('show');
    });
}

function saveAirline() {
    const oldCode = $('#airlineCodeOld').val();
    const airlineData = {
        airline_code: $('#airlineCodeInput').val().toUpperCase(),
        airline_name: $('#airlineName').val(),
        country: $('#airlineCountry').val()
    };
    
    if (oldCode) {
        $.ajax({
            url: '/api/airlines/' + oldCode,
            method: 'PUT',
            contentType: 'application/json',
            data: JSON.stringify(airlineData),
            success: function() {
                $('#airlineModal').modal('hide');
                airlinesTable.ajax.reload();
                loadReferenceData();
                showToast('Airline updated successfully!', 'success');
            },
            error: function() {
                showToast('Error updating airline', 'error');
            }
        });
    } else {
        $.ajax({
            url: '/api/airlines',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(airlineData),
            success: function() {
                $('#airlineModal').modal('hide');
                airlinesTable.ajax.reload();
                loadReferenceData();
                showToast('Airline added successfully!', 'success');
            },
            error: function() {
                showToast('Error adding airline', 'error');
            }
        });
    }
}

function deleteAirline(code) {
    if (confirm('Are you sure you want to delete this airline?')) {
        $.ajax({
            url: '/api/airlines/' + code,
            method: 'DELETE',
            success: function() {
                airlinesTable.ajax.reload();
                loadReferenceData();
                showToast('Airline deleted successfully!', 'success');
            },
            error: function() {
                showToast('Error deleting airline', 'error');
            }
        });
    }
}

// Aircraft functions
function initAircraftTable() {
    aircraftTable = $('#aircraftTable').DataTable({
        ajax: {
            url: '/api/aircraft',
            dataSrc: ''
        },
        columns: [
            { data: 'aircraft_id' },
            { data: 'aircraft_type' },
            { data: 'manufacturer' },
            { data: 'capacity' },
            { data: 'range_km' },
            {
                data: null,
                render: function(data) {
                    return `
                        <div class="table-actions">
                            <button class="btn btn-sm btn-warning" onclick="editAircraft(${data.aircraft_id})">Edit</button>
                            <button class="btn btn-sm btn-danger" onclick="deleteAircraft(${data.aircraft_id})">Delete</button>
                        </div>
                    `;
                }
            }
        ]
    });
}

function openAircraftForm() {
    $('#aircraftModalTitle').text('Add Aircraft');
    $('#aircraftForm')[0].reset();
    $('#aircraftId').val('');
}

function editAircraft(id) {
    $.get('/api/aircraft/' + id, function(aircraft) {
        $('#aircraftModalTitle').text('Edit Aircraft');
        $('#aircraftId').val(aircraft.aircraft_id);
        $('#aircraftTypeInput').val(aircraft.aircraft_type);
        $('#manufacturer').val(aircraft.manufacturer);
        $('#capacity').val(aircraft.capacity);
        $('#rangeKm').val(aircraft.range_km);
        
        $('#aircraftModal').modal('show');
    });
}

function saveAircraft() {
    const aircraftId = $('#aircraftId').val();
    const aircraftData = {
        aircraft_type: $('#aircraftTypeInput').val(),
        manufacturer: $('#manufacturer').val(),
        capacity: parseInt($('#capacity').val()),
        range_km: parseInt($('#rangeKm').val())
    };
    
    if (aircraftId) {
        aircraftData.aircraft_id = parseInt(aircraftId);
        $.ajax({
            url: '/api/aircraft/' + aircraftId,
            method: 'PUT',
            contentType: 'application/json',
            data: JSON.stringify(aircraftData),
            success: function() {
                $('#aircraftModal').modal('hide');
                aircraftTable.ajax.reload();
                showToast('Aircraft updated successfully!', 'success');
            },
            error: function() {
                showToast('Error updating aircraft', 'error');
            }
        });
    } else {
        $.ajax({
            url: '/api/aircraft',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(aircraftData),
            success: function() {
                $('#aircraftModal').modal('hide');
                aircraftTable.ajax.reload();
                showToast('Aircraft added successfully!', 'success');
            },
            error: function() {
                showToast('Error adding aircraft', 'error');
            }
        });
    }
}

function deleteAircraft(id) {
    if (confirm('Are you sure you want to delete this aircraft?')) {
        $.ajax({
            url: '/api/aircraft/' + id,
            method: 'DELETE',
            success: function() {
                aircraftTable.ajax.reload();
                showToast('Aircraft deleted successfully!', 'success');
            },
            error: function() {
                showToast('Error deleting aircraft', 'error');
            }
        });
    }
}

